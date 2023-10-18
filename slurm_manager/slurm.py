import subprocess
import sqlite3
from pathlib import Path
import re
import pandas as pd
import logging

class SlurmManager:
    """
    A manager class for interfacing with the Slurm workload manager.
    
    Attributes:
    - parent_dir (str): The parent directory where the manager operates.
    - DB_PATH (Path): The path to the SQLite database where job information is stored.
    
    Methods:
    - init_db(): Initializes the SQLite database for job storage.
    - insert_job(): Inserts a new job entry into the database.
    - parse_scontrol_output(): Parses the output of the 'scontrol' command.
    - submit_job(): Submits a job to the Slurm manager.
    - job_table(): Retrieves a DataFrame of jobs, filtered by optional status.
    - cancel_job(): Cancels a job in the Slurm manager.
    - refresh_job_status(): Updates the status of jobs in the database.
    """

    def __init__(self, parent_dir: str):
        """Initializes the SlurmManager with the specified parent directory."""
        self.parent_dir = Path(parent_dir).absolute()
        self.DB_PATH = Path.home() / "slurm_jobs.db"
        self._init_loggers()
        self.init_db()

    def _init_loggers(self):
        """Private method to initialize local and global loggers."""
        # Local logger
        local_log_file_path = self.parent_dir / ".slurm.log"
        self.local_logger = self._configure_logger(f"SlurmManager_Local_{self.parent_dir.name}", local_log_file_path)

        # Global logger
        global_log_file_path = Path.home() / ".global_slurm.log"
        self.global_logger = self._configure_logger("SlurmManager_Global", global_log_file_path, stream=False)

    def _configure_logger(self, name, log_file_path, stream=True):
        """Private method to configure and return a logger."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # FileHandler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # StreamHandler (prints to console), if required
        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def init_db(self):
        """Initializes the SQLite database."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs
                (id INTEGER PRIMARY KEY, parent_dir TEXT, jobid TEXT, jobname TEXT, status TEXT, workingdir TEXT)
            ''')

    def insert_job(self, jobid: str, db_jobname: str, status: str, workingdir: str):
        """Inserts a new job entry into the database."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("INSERT INTO jobs (parent_dir, jobid, jobname, status, workingdir) VALUES (?, ?, ?, ?, ?)",
                        (str(self.parent_dir), jobid, db_jobname, status, workingdir))

    def parse_scontrol_output(self, output: str) -> dict:
        """
        Parses the output of the 'scontrol' command to extract job details.
        
        Args:
        - output (str): The stdout string from the 'scontrol' command.

        Returns:
        - dict: A dictionary containing details of the job.
        """
        return {
            "jobid": re.search(r'JobId=(\d+)', output).group(1),
            "status": re.search(r'JobState=([^\s]+)', output).group(1),
            "workingdir": re.search(r'WorkDir=([^\s]+)', output).group(1)
        }

    def submit_job(self, path: str, db_jobname: str, resubmit: bool = False):
        """
        Submits a job to the Slurm workload manager.
        
        Args:
        - path (str): The path to the job script.
        - db_jobname (str): The name to be used for the job in the database.
        - resubmit (bool, optional): Whether to resubmit the job if it's already running or pending. Defaults to False.
        """
        abs_path = Path(path).absolute()
        
        # Checking existing jobs
        existing_jobs = self.job_table()
        matching_jobs = existing_jobs[existing_jobs['workingdir'] == str(abs_path)]

        # If jobs found, act accordingly
        if not matching_jobs.empty:
            active_jobs = matching_jobs[matching_jobs['status'].isin(["RUNNING", "PENDING"])]
            if not active_jobs.empty:
                if not resubmit:
                    self.local_logger.warning(f"A job from working directory {abs_path} is already {active_jobs.iloc[0]['status']}")
                    return
            elif any(matching_jobs['status'] == "CANCELLED"):
                self.local_logger.warning(f"A job from working directory {abs_path} was previously cancelled")

        # Submitting job to Slurm
        result = subprocess.run(['sbatch', 'submit.sh'], cwd=abs_path, capture_output=True, text=True)
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        jobid = match.group(1) if match else None

        scontrol_result = subprocess.run(['scontrol', '-dd', 'show', 'job', jobid], capture_output=True, text=True)
        job_details = self.parse_scontrol_output(scontrol_result.stdout)

        self.insert_job(job_details["jobid"], db_jobname, job_details["status"], job_details["workingdir"])
        self.local_logger.info(f"Submitted job with JobID: {jobid}, JobName for DB: {db_jobname}")
        self.global_logger.info(f"Submitted job with JobID: {jobid} in directory: {abs_path}")

    def job_table(self, status: str = None) -> pd.DataFrame:
        """
        Retrieves a DataFrame of jobs from the database, optionally filtered by status.
        
        Args:
        - status (str, optional): The status to filter the jobs by. If not specified, all jobs are retrieved.

        Returns:
        - pd.DataFrame: A DataFrame of jobs.
        """
        self.refresh_job_status()
        query = f"SELECT * FROM jobs WHERE parent_dir='{str(self.parent_dir)}'"
        if status:
            query += f" AND status='{status.upper()}'"
        with sqlite3.connect(self.DB_PATH) as conn:
            return pd.read_sql_query(query, conn)

    def cancel_job(self, slurm_jobid: str):
        """
        Cancels a job in the Slurm workload manager.
        
        Args:
        - slurm_jobid (str): The job ID to cancel.
        """
        result = subprocess.run(['scancel', str(slurm_jobid)], capture_output=True, text=True)
        if result.returncode == 0:
            self.local_logger.info(f"Cancelled SLURM job with SlurmJobID: {slurm_jobid}")
            self.global_logger.info(f"Cancelled SLURM job with SlurmJobID: {slurm_jobid} from directory: {self.parent_dir}")
        else:
            self.local_logger.error(f"Failed to cancel SLURM job with SlurmJobID: {slurm_jobid}. Error: {result.stderr}")
            self.global_logger.error(f"Failed to cancel SLURM job with SlurmJobID: {slurm_jobid} from directory: {self.parent_dir}. Error: {result.stderr}")

    def refresh_job_status(self):
        """
        Updates the status of jobs in the SQLite database based on their current status in Slurm.
        """
        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT jobid FROM jobs WHERE parent_dir='{self.parent_dir}'")
            jobids = cursor.fetchall()

            for (jobid,) in jobids:
                # Use scontrol to get the job details
                result = subprocess.run(['scontrol', '-dd', 'show', 'job', jobid], capture_output=True, text=True)
                if "Invalid job id specified" in result.stderr:
                    # Set status to UNKNOWN if job ID is not recognized by Slurm
                    new_status = "UNKNOWN"
                else:
                    job_details = self.parse_scontrol_output(result.stdout)
                    new_status = job_details["status"]

                # Update the job status in the SQLite database
                cursor.execute("UPDATE jobs SET status=? WHERE jobid=? AND parent_dir=?", (new_status, jobid, str(self.parent_dir)))

            conn.commit()
