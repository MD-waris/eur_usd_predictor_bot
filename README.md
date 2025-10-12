# EURUSD Automation Template

Files in this zip:
- eurusd_predictor.py  (converted from your notebook)
- .github/workflows/run.yml  (GitHub Actions workflow to run hourly)
- requirements.txt

Instructions:
1. Create a new GitHub repository (or use an existing one).
2. Upload all files from this zip to the root of your repository (preserve the .github/workflows path).
3. In your repository settings -> Secrets -> Actions, add two secrets:
   - TELEGRAM_TOKEN : your Telegram bot token
   - CHAT_ID : your Telegram chat ID
4. Commit to the default branch. Go to the Actions tab and run the workflow manually to test.

Notes:
- The script captures notebook stdout and sends the last printed line as the prediction.
- I did not change your algorithm logic; I only wrapped stdout capture and added a small Telegram sender at the end.
- If your notebook requires additional packages, add them to requirements.txt or modify the workflow to install them.

Generated at: 2025-10-12T01:21:01.429289Z
