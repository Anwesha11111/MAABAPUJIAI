@echo off
REM ══════════════════════════════════════════════════════════════════
REM  deploy_to_hf.bat  —  Deploy to Anwesha11111/BapujiAI
REM  Run this from your project folder:
REM    C:\Users\DELL\Downloads\New folder\PROJECTS\files (3)\
REM ══════════════════════════════════════════════════════════════════

echo.
echo  🌌  Deploying Life in the Multiverse Guide
echo       → https://huggingface.co/spaces/Anwesha11111/BapujiAI
echo.

REM ── 1. Init git if needed ─────────────────────────────────────────
if not exist ".git" (
    git init
    echo [OK] git init
)

REM ── 2. Set up Git LFS for large files ────────────────────────────
git lfs install
echo *.json filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo *.npy filter=lfs diff=lfs merge=lfs -text >> .gitattributes
echo [OK] git lfs ready

REM ── 3. Add HF remote ──────────────────────────────────────────────
git remote remove hf 2>nul
git remote add hf https://huggingface.co/spaces/Anwesha11111/BapujiAI
echo [OK] remote set to Anwesha11111/BapujiAI

REM ── 4. Stage all files ────────────────────────────────────────────
git add app.py requirements.txt book_chunks.json README.md .gitattributes
echo [OK] files staged

REM ── 5. Commit ─────────────────────────────────────────────────────
git commit -m "🌌 Life in the Multiverse Guide — 820 real book chunks (276-page OCR)"
echo [OK] committed

REM ── 6. Push → triggers HF auto-build ─────────────────────────────
echo.
echo  Pushing to HF Spaces (may ask for your HF username + token)...
echo  Username: Anwesha11111
echo  Password: your HF token from https://huggingface.co/settings/tokens
echo.
git push hf main --force

echo.
echo  ✅  Done! Your Space will be live in ~3-5 minutes:
echo      https://huggingface.co/spaces/Anwesha11111/BapujiAI
echo.
echo  🙏  Jai Sat Chit Anand!
pause
