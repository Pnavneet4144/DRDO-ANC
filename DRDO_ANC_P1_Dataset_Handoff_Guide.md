# DRDO-ANC Dataset Handoff Guide for P1

This guide explains how to package the current prototype dataset and share it with P1.

> **Important — prototype only:** This dataset was generated using LibriSpeech `dev-clean`. The project plan reserves `dev-clean` for testing, so this package is **not the final leakage-safe training dataset**. P1 should treat it as a prototype for inspecting the manifest/audio pipeline only, and should not use it for final training or evaluation claims.

## 1. What you are sharing

The ZIP should contain:

- `master_training.csv` — manifest listing each example and its clean, noise, reference, and mixed audio paths.
- `generated/clean/` — clean speech clips.
- `generated/noise/` — noise clips used for the mixtures.
- `generated/mixed/` — noisy mixture clips.
- `generated/reference/` — simulated reference-microphone clips.

The audio is stored outside Git intentionally. Do not add the generated WAV files to the repository.

## 2. Check that the files exist

Open **PowerShell** and change to your repository folder:

```powershell
cd D:\DRDO-ANC
```

Check the manifest and generated folders:

```powershell
Test-Path .\ai_engine\data\master_training.csv
Test-Path .\ai_engine\data\generated\clean
Test-Path .\ai_engine\data\generated\noise
Test-Path .\ai_engine\data\generated\mixed
Test-Path .\ai_engine\data\generated\reference
```

Each command should print `True`. If any prints `False`, stop and resolve the missing path before creating the ZIP.

## 3. Create the ZIP

Run this in PowerShell from `D:\DRDO-ANC`:

```powershell
$zip = "$env:USERPROFILE\Desktop\ANC_dataset_for_P1.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive `
  -Path .\ai_engine\data\master_training.csv, .\ai_engine\data\generated `
  -DestinationPath $zip `
  -CompressionLevel Fastest
```

Wait for PowerShell to finish. A ZIP with thousands of audio files may take a while to create and may be large.

## 4. Verify the ZIP

Run:

```powershell
Test-Path "$env:USERPROFILE\Desktop\ANC_dataset_for_P1.zip"
Get-Item "$env:USERPROFILE\Desktop\ANC_dataset_for_P1.zip" |
  Select-Object Name, Length
```

The first command should print `True`. The second displays the ZIP filename and size in bytes.

Optionally, inspect its contents:

```powershell
tar -tf "$env:USERPROFILE\Desktop\ANC_dataset_for_P1.zip" | Select-Object -First 20
```

Confirm the archive contains `master_training.csv` and the `generated` folders. The archive may include a leading `ai_engine/data/` path; if so, P1 should extract it into the repository root so the original paths are preserved.

## 5. Send it to P1 on WhatsApp

1. Open WhatsApp Web/Desktop or the phone app.
2. Open the direct chat with P1.
3. Attach `ANC_dataset_for_P1.zip` as a **Document/File**, not as photos or videos.
4. Wait for the upload to finish, then send it.
5. If WhatsApp refuses the file because it is too large, do not split it into arbitrary chunks. Upload it to an approved shared drive (for example, your team's Google Drive) and send P1 the access link instead.

Before sending, check the ZIP size. WhatsApp's upload limit can vary by client/version and may be too low for a dataset with thousands of WAV files.

## 6. Instructions for P1 after receiving it

Send P1 these instructions along with the ZIP:

1. Save the ZIP locally.
2. Extract it into a temporary folder first and inspect the contents.
3. If the archive paths begin with `ai_engine/data/`, extract into the repository root. The resulting files should be at:
   - `ai_engine/data/master_training.csv`
   - `ai_engine/data/generated/clean/`
   - `ai_engine/data/generated/noise/`
   - `ai_engine/data/generated/mixed/`
   - `ai_engine/data/generated/reference/`
4. Do not commit the generated audio files to Git.
5. Treat the dataset as a **prototype only**, because it uses `dev-clean`, which is reserved for testing in the project plan.
6. If the manifest paths do not resolve after extraction, tell the dataset owner before changing the CSV paths.

## 7. Troubleshooting

- **A `Test-Path` command prints `False`:** Check the spelling and exact location of the manifest/folder.
- **The ZIP is too large for WhatsApp:** Use an approved shared drive and send a link with the necessary access permissions.
- **The archive seems to have the wrong folder structure:** Run `tar -tf` to inspect its paths before sending. Preserve the `ai_engine/data/` directory structure when extracting.
- **P1 cannot open the audio:** Check that the ZIP extracted completely and that the referenced paths exist. The manifest alone does not contain the audio data.

## 8. Final checklist

- [ ] `master_training.csv` is included.
- [ ] All four generated audio folders are included.
- [ ] The ZIP was created and its size checked.
- [ ] The archive paths were inspected.
- [ ] P1 was told this is a prototype, not the final leakage-safe training set.
- [ ] No generated audio was committed to Git.
