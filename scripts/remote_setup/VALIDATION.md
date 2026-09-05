# Validation evidence

No packages were installed, no GPU was initialized, and existing results were not touched.

| Target | Python / platform | Resolved packages |
|---|---|---|
| geo | 3.12 / win_amd64 | 33 |
| sam | 3.12 / win_amd64 | 58 |
| lama | 3.11 / win_amd64 | 18 |

Used pip 25.2: `pip install --dry-run --ignore-installed --only-binary=:all: --platform win_amd64 --python-version VERSION --extra-index-url https://download.pytorch.org/whl/cu128 -r REQUIREMENTS --report REPORT`.

LaMa additionally used `--find-links` with a temporary wheel directory. Its required `fire==0.5.0` was built from the official source using `pip wheel --no-deps --no-build-isolation`; no package was installed. Remote installation permits source only for fire.

Official LaMa metadata: https://pypi.org/pypi/simple-lama-inpainting/0.1.2/json. NumPy >=1.24.3,<2 and Pillow >=9.5,<10 conflict with the former shared environment. Transformers 5.13 vision requires Pillow >=10.0.1, so LaMa now has a separate Python 3.11 environment.

Passed: PowerShell parsing/static checks; common-helper regressions (unchanged/drift/missing/failed freezes, native failure propagation, absent launcher Python); temporary Git integration fixtures (absent/bootstrap-only/empty root, wrong origin, detached/wrong branch, non-Git user data, dirty refusal/override); 20 subprocess smoke scenarios covering successful mocked CUDA calls and bad direction, nondeterminism, non-finite scores/masks, wrong revision, absent CUDA/masks, wrong dimensions, empty/full masks, and forward/load exceptions.

The two added output regressions first failed: LaMa crop concealed wrong dimensions, and SAM3 boolean conversion concealed NaN masks. Both now reject invalid outputs.

Runtime limits: dry-run resolution is not installed import/ABI/CUDA verification. Installer runs pip check, real application imports, version checks and full freezes on the remote machine. Real GeoRanker/LaMa/SAM3 CUDA smoke remains required after deployment. Python 3.12 and 3.11 registrations, drivers, remote access, inputs/caches, model downloads and SAM3 license/auth are outstanding prerequisites.

Live pin verification: both Hugging Face revision API responses matched their configured commit IDs. Streamed official adapter bytes matched the existing weight SHA-256. Official adapter_config bytes revealed and corrected an incorrect hash; the verified SHA-256 is `48910F66A263A4E5146E98F0E14ACF5421454BCA01C9A3580C0E423D69A56565`. No downloaded model bytes were saved or loaded. The prefetch reference regression also verifies its on-disk atomic reference update against a fake Hub cache.

## Windows PowerShell 5.1 correction

Reproduced the original Test-SyncRepository failure at `git clone --bare ... 2>$null` with NativeCommandError under Windows PowerShell 5.1.26100.9168 / Desktop. A separate native-command regression reproduced the same failure in the old Invoke-Checked when its caller redirected stderr. The helper now treats redirected native stderr as diagnostics on the host stream, preserves stdout separately, and throws on nonzero exit. The preference override is function-local and restored in finally; command lookup still fails explicitly.

Exact Python test interpreter: `C:\Users\phdwf\miniconda3\python.exe`, Python 3.13.9 (Anaconda), Pillow 12.1.1, NumPy 2.4.2. Executed directly with `test_smoke_mocked.py` (unittest, not pytest): 4 tests passed, including 20 subprocess scenarios. No Python 3.14 or dependency-free portability claim is made. No packages were installed or existing environments changed in this correction.
