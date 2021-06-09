"""
to cope with DNS spoofing, automatically retry a git command until success
"""

import os, argparse, warnings, time
from typing import NoReturn


_CMD = {
	"submodule": "git submodule update --remote --recursive --merge",
	"push": "git push origin --all",
	"fetch": "git fetch --all",
}


def get_parser() -> dict:
    """
    """
    description = "predefined and custom git operations, in order to cope with DNS spoofing"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--submodule", action="store_true",
        help=f"submodule update",
        dest="submodule",
    )
    parser.add_argument(
        "-f", "--fetch", action="store_true",
        help=f"fetch origin",
        dest="fetch",
    )
    parser.add_argument(
        "-p", "--push", action="store_true",
        help=f"push origin",
        dest="push",
    )
    parser.add_argument(
        "-c", "--custom", type=str,
        help=f"custom git command, enclosed within \"",
        dest="custom",
    )

    args = vars(parser.parse_args())

    return args


def run(action:str) -> NoReturn:
    """ finished, checked,

    Parameters
    ----------
    action: str,
        git command (action, operation) to be executed
    """
    cmd = _CMD.get(action.lower(), action.lower())
    ret_code = None
    n_iter = 0
    while ret_code != 0:
        if n_iter > 0:
            print(f"""retry {n_iter} {"times" if n_iter > 1 else "time"}""")
            time.sleep(1)
        ret_code = os.system(cmd)
        n_iter += 1
    print("Execution success")
	

if __name__ == "__main__":
    args = get_parser()
    custom = args.get("custom", None)
    args = [k for k,v in args.items() if v and k not in ["custom",]]
    if custom:
        if len(args) > 0:
            warnings.warn("custom command is given, additional arguments are ignored!")
        run(custom)
        exit(0)
    if len(args) != 1:
        raise ValueError("one and only one predefined command can be executed!")
    run(args[0])
