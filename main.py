import subprocess
import platform


def start_ansys_workbench(executable_path=None):
    """Start ANSYS 2024 R2 Workbench.

    Parameters
    ----------
    executable_path : str, optional
        Path to the Workbench executable. If not provided, a default
        location is used based on the operating system.
    """
    if executable_path is None:
        if platform.system() == "Windows":
            executable_path = r"C:\\Program Files\\ANSYS Inc\\v242\\Framework\\bin\\win64\\RunWB2.exe"
        else:
            executable_path = "runwb2"

    try:
        subprocess.Popen([executable_path])
        print("ANSYS Workbench started.")
    except FileNotFoundError:
        print(f"Executable not found: {executable_path}")
    except Exception as e:
        print(f"Failed to start ANSYS Workbench: {e}")


if __name__ == "__main__":
    start_ansys_workbench()
