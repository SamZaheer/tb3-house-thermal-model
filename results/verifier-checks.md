# Verifier checks run locally (no agent involved)

Run 2026-10-26 to 2026-10-28 on macOS arm64, Docker 29.8 on Colima (8 CPU, 12 GB), Harbor 0.23.1.dev202609170426.

| Check | Command | Result |
|---|---|---|
| Static checks | `for c in scripts/checks/check-*.sh; do bash $c tasks/house-thermal-model; done` (scripts from `harbor-framework/terminal-bench` at `1dcda871`) | 26/26 pass ([logs](checks/static/)) |
| Docker build + oracle | `harbor run -p tasks/house-thermal-model --agent oracle --env docker` | reward 1.0; winter 0.995 and spring 0.982 of day-features in tolerance ([output](execution/oracle/verifier/test-stdout.txt)) |
| Nop | `harbor run -p tasks/house-thermal-model --agent nop --env docker` | reward 0.0; the stub's flat forecast scores 0.255 (winter) and 0.290 (spring) ([output](execution/nop/verifier/test-stdout.txt)) |
| Artifact collection | the oracle and nop trials above; Harbor's `tar` command with the same excludes run in the agent image on an `/app` holding an extra `data/fitted_params.json` | Harbor collected `/app` without the three given data files; the extra file was kept; the verifier put the given files back and ran the submission |
| Sandbox: cheating predictor | verifier image, a `predict.py` that tries to list and read `/tests/hidden/truth_*.csv` and to write `/logs/verifier/reward.txt` ([script](exploit-test/predict.py), [output](exploit-test/output.txt)) | reward 0; `/tests` unreadable as `nobody` (`Permission denied`), reward write `Permission denied` |
| Sandbox: cheating predictor that leaves a background process | verifier image, a `predict.py` that detaches a process which keeps trying to write the reward ([script](exploit-test/daemon/predict.py), [output](exploit-test/daemon/output.txt)) | reward 0; nothing left running as `nobody` after the run |
| Sandbox: output linked to the truth | verifier image, a `predict.py` that makes `OUT.csv` a symbolic link to `/tests/hidden/truth_<period>.csv` ([script](exploit-test/symlink/predict.py), [output](exploit-test/symlink/output.txt)); a second one that tries a hard link ([script](exploit-test/hardlink/predict.py), [output](exploit-test/hardlink/output.txt)) | reward 0 each: the symbolic link is rejected (not a regular file owned by `nobody`), the hard link is refused (`Permission denied`) |
| Sandbox: predictor that runs out of memory | verifier image under `--memory 4g`, a `predict.py` that allocates memory until the limit is hit ([script](exploit-test/oom/predict.py), [output](exploit-test/oom/output.txt)) | reward 0; the kernel killed the predictor (exit -9) and the grader finished; no memory cap of the verifier's own is involved |
| Honest output variants | verifier image, the reference wrapped five ways: a file kept in `/app/data/`, NaN on the spin-up days, times with a `-06:00` offset, times in UTC, `OUT.csv` a symbolic link to the predictor's own file ([scripts and results](output-variants/)) | reward 1 each, with the reference's shares (0.995 / 0.982) |
| Truth regeneration | `python3 tools/truth/make_truth.py OUT_DIR` with numba on macOS arm64, and in the agent image (Linux, no numba) | macOS: all six data files byte-identical to the packaged ones. Linux image: the meter and true series byte-identical; the three weather files agree to within 0.01, one unit of their last rounded digit |
| Verifier network | `harbor run` with a scratch copy of the task whose `test.sh` first tries to fetch `http://example.com`, with and without `network_mode = "no-network"` | with it: the fetch failed (`RemoteDisconnected`; Harbor's egress control denies all traffic during verification); without it (control): the fetch succeeded. The probe copy was not kept |
| The three imported Harbor trial submissions: Claude Code / Opus 5.5 (max), in the packaged verifier | verifier image with each submission copied to `/app` | reward 0 each, failing both winter and spring; the same shares as the imported submissions re-scored locally with the same numpy, scipy and pandas ([outputs](trials/opus-5.5/)) |
| The three imported Harbor trial submissions: Claude Code / GPT-6.1 Sol (xhigh), in the packaged verifier | the same, with each delivered `predict.py` alone copied to `/app` | reward 0 each, passing winter and failing spring; the same shares as the grading of each trial's full final workspace ([outputs](trials/gpt-6.1-sol/)) |

Commands for the re-scores and exploit tests (from the repo root):

```bash
docker build -t hphouse-verifier tasks/house-thermal-model/tests
docker run --rm --network none --cpus 2 --memory 4g -v "$PWD/<dir-with-predict.py>":/sub:ro hphouse-verifier \
  bash -c 'cp -r /sub/. /app/ && /tests/test.sh; cat /logs/verifier/reward.txt'
```
