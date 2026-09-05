@{
    # Copy this file to Config.psd1 on each machine. Config.psd1 is intentionally
    # excluded by the scripts in this directory and must never contain tokens.
    MachineRole  = 'remote-worker'
    WorkerName   = 'remote4090'
    ShardIndex   = 0
    NumShards    = 1

    ProjectRoot  = 'C:\Users\xx0073.UNT\GeoBayes'
    RepoUrl      = 'https://github.com/Kerwinxxp/GeomPL.git'
    Branch       = 'main'
    PythonVersion = '3.12'

    RequiredGpuPattern = 'RTX 4090'
    MinimumFreeDiskGB  = 40
}
