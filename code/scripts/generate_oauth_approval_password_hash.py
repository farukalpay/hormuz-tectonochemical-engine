from __future__ import annotations

import argparse
import getpass

from hte.oauth import hash_approval_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTE OAuth approval password hash")
    parser.add_argument("--password", help="Approval password (omit to prompt securely)")
    args = parser.parse_args()

    password = args.password or getpass.getpass("Approval password: ")
    if not password:
        raise SystemExit("Password cannot be empty")

    print(hash_approval_password(password))


if __name__ == "__main__":
    main()
