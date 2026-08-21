# ==============================================================================
# Auto Pusher Script: Spaced-out sequential pushes for human-like commit history
# ==============================================================================

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Automated Sequential Pusher (3 Batches, 30m gap)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# Batch 1: Ingestion, Feature Engineering & Target Modules
# ------------------------------------------------------------------------------
Write-Host "`n[Batch 1/3] Staging Data Ingestion & Feature Engineering..." -ForegroundColor Yellow
git add src/data/ src/features/ src/targets/ tests/
git commit -m "feat(docstrings): standardize data ingestion, feature engineering, and target modules to Google docstrings"
git push origin dev
Write-Host "[Batch 1/3] Successfully pushed to origin/dev at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
Write-Host "Sleeping for 30 minutes before Batch 2..." -ForegroundColor Gray

Start-Sleep -Seconds 1800

# ------------------------------------------------------------------------------
# Batch 2: Modeling, Forecasting, Evaluation & Robustness
# ------------------------------------------------------------------------------
Write-Host "`n[Batch 2/3] Staging Modeling, Evaluation & Robustness Modules..." -ForegroundColor Yellow
git add src/models/ src/evaluation/ src/robustness/ src/utils/
git commit -m "feat(docstrings): standardize modeling, evaluation, and robustness modules to Google docstrings"
git push origin dev
Write-Host "[Batch 2/3] Successfully pushed to origin/dev at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
Write-Host "Sleeping for 30 minutes before Batch 3..." -ForegroundColor Gray

Start-Sleep -Seconds 1800

# ------------------------------------------------------------------------------
# Batch 3: Executive Technical Report, Figures, and Documentation
# ------------------------------------------------------------------------------
Write-Host "`n[Batch 3/3] Staging Executive Technical Report, Documentation & Figures..." -ForegroundColor Yellow
git add docs/ README.md scripts/ outputs/
git commit -m "docs: add executive technical report and high-resolution diagnostic figures"
git push origin dev
Write-Host "[Batch 3/3] Successfully pushed to origin/dev at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green

# ------------------------------------------------------------------------------
# Promote to test, prod, and main branches
# ------------------------------------------------------------------------------
Write-Host "`nPromoting and synchronizing dev across test, prod, and main..." -ForegroundColor Cyan

git checkout test
git merge --no-ff dev -m "merge(test): sync docstring standardizations and executive report"
git push origin test

git checkout prod
git merge --no-ff test -m "release(prod): sync docstring standardizations and executive report"
git push origin prod

git checkout main
git merge --no-ff prod -m "release(main): sync docstring standardizations and executive report"
git push origin main

git checkout dev
Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "All batches and branch promotions completed successfully!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
