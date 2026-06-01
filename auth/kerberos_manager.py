"""
Kerberos Authentication Manager

Handles Kerberos ticket operations using subprocess calls.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path


class KerberosManager:
    """Handles Kerberos ticket operations using subprocess calls."""

    def check_existing_ticket(self) -> bool:
        """Check if a valid Kerberos ticket already exists."""
        try:
            # Try klist -s first (silent mode, returns non-zero if no valid tickets)
            result = subprocess.run(
                ['klist', '-s'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode == 0:
                return True

            # Fallback to verbose check if klist -s is not supported
            klist_result = subprocess.run(
                ['klist'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if klist_result.returncode == 0:
                output = klist_result.stdout
                stderr = klist_result.stderr

                has_ticket = (
                    "Default principal:" in output or "krbtgt" in output or
                    "Valid starting" in output or "Ticket cache:" in output
                )

                no_ticket_indicators = [
                    "No credentials cache found",
                    "Credentials cache file does not exist",
                    "No credentials",
                    "klist: No credentials cache found"
                ]

                has_error = any(
                    indicator in output or indicator in stderr
                    for indicator in no_ticket_indicators
                )

                if has_ticket and not has_error:
                    return True
            return False
        except FileNotFoundError:
            print(
                "[ERROR] 'klist' command not found. Please ensure Kerberos tools are installed.",
                file=sys.stderr
            )
            return False
        except Exception as e:
            print(f"[ERROR] Error checking existing ticket: {str(e)}", file=sys.stderr)
            return False

    def destroy_existing_ticket(self) -> bool:
        """Destroy existing Kerberos ticket."""
        try:
            subprocess.run(
                ['kdestroy'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            print("[INFO] Existing Kerberos ticket destroyed.")
            return True
        except Exception as e:
            print(f"[ERROR] Error destroying ticket: {str(e)}", file=sys.stderr)
            return False

    @staticmethod
    def _get_cache_env():
        """Return (cache_file, env_dict) for the file-based Kerberos credential cache."""
        safe_username = os.environ.get('USERNAME', getpass.getuser()).replace('/', '_').replace('\\', '_')
        cache_file = Path.home() / f"krb5cc_{safe_username}"
        env = os.environ.copy()
        env['KRB5CCNAME'] = f"FILE:{cache_file}"
        return cache_file, env

    def renew_ticket(self) -> bool:
        """Attempt to silently renew the Kerberos ticket using kinit -R.

        Returns True if renewal succeeded, False otherwise.
        """
        _, env = self._get_cache_env()
        try:
            result = subprocess.run(
                ['kinit', '-R'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
            if result.returncode == 0:
                print("[SUCCESS] Kerberos ticket renewed automatically.")
                return True
        except Exception as e:
            print(f"[DEBUG] kinit -R exception: {e}")
        print("[WARNING] Automatic Kerberos renewal failed — ticket may need manual re-authentication.")
        return False

    def get_kerberos_ticket(self, force_new: bool = False) -> bool:
        """Get a new Kerberos ticket."""
        username = f"{os.environ.get('USERNAME', getpass.getuser())}@FNAL.GOV"
        print(f"[INFO] Attempting to authenticate as: {username}")

        cache_file, env = self._get_cache_env()

        os.environ['KRB5CCNAME'] = env['KRB5CCNAME']

        if force_new:
            self.destroy_existing_ticket()

        try:
            print(
                "\n[INFO] The application will now request a Kerberos ticket.\n"
                "Please enter your password in the terminal/console window where the script was launched.\n"
            )
            print(f"[INFO] Using credential cache: {cache_file}\n")

            kinit_process = subprocess.run(
                ['kinit', username],
                check=True,
                env=env,
                stderr=subprocess.PIPE,
                timeout=60
            )

            if kinit_process.returncode == 0:
                print(f"[SUCCESS] Successfully obtained Kerberos ticket for {username}")
                subprocess.run(['klist'], env=env)
                return True
            return False
        except subprocess.TimeoutExpired:
            print("[ERROR] kinit command timed out after 60 seconds.", file=sys.stderr)
            return False
        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] Authentication failed with error code: {e.returncode}.",
                file=sys.stderr
            )
            if e.stderr:
                stderr_text = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='replace')
                print(f"[ERROR] Details: {stderr_text}", file=sys.stderr)
            return False
        except FileNotFoundError:
            print(
                "[ERROR] 'kinit' command not found. Please ensure Kerberos tools are installed.",
                file=sys.stderr
            )
            return False
        except Exception as e:
            print(
                f"[ERROR] An unexpected error occurred during kinit: {str(e)}",
                file=sys.stderr
            )
            return False
