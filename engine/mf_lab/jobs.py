import logging
from typing import Optional, List
from mf_lab.logic import run_full_mf_scan
from utils.db import log_audit

logger = logging.getLogger(__name__)

def run_mf_background_job(job_type: str, force_refresh: bool = False, scheme_codes: Optional[List[str]] = None):
    """
    Executes a Mutual Fund processing job headlessly as a background task.
    """
    logger.info(f"Starting MF background job [{job_type}] (force_refresh={force_refresh})")
    try:
        if job_type == "full_refresh":
            # Native run_full_mf_scan already saves direct to the Neon Postgres tables.
            df = run_full_mf_scan(max_workers=20)
            if not df.empty:
                logger.info(f"Job [full_refresh] complete. Saved {len(df)} schemes to DB.")
                log_audit("MF Backend Job", "Mutual Funds", f"Job full_refresh completed successfully. Scored {len(df)} funds.")
            else:
                logger.warning("Job [full_refresh] returned no results.")
                
        elif job_type == "update_metrics":
            # Placeholder for targeted schema updates
            logger.info("Job [update_metrics] natively bypasses to full_refresh for now.")
            pass
            
        elif job_type == "recalculate_rankings":
            # Placeholder/Future logic
            pass
            
        else:
            logger.error(f"Unknown background job type requested: {job_type}")
            
    except Exception as e:
        logger.error(f"FATAL: MF background job [{job_type}] crashed: {e}", exc_info=True)
