# Email Alert System

The AI Security System includes an email alert feature that automatically sends notifications when a person is detected in the video stream.

## Configuration

Email alerts are configured using environment variables in the `.env` file:

```env
FROM_EMAIL=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
TO_EMAIL=recipient@gmail.com
```

### Setting up Gmail App Password

For Gmail accounts, you'll need to use an App Password instead of your regular password:

1. Go to your Google Account settings
2. Select Security → 2-Step Verification
3. Under "Signing in to Google," select App passwords
4. Select "Mail" and generate a password
5. Use this generated password as `EMAIL_PASSWORD`

## How it Works

1. **Person Detection**: When the YOLO model detects a person in the video stream
2. **Cooldown Check**: Ensures emails aren't sent too frequently (5-minute cooldown by default)
3. **Email Composition**: Creates an HTML email with:
   - Timestamp of detection
   - Detection confidence level
   - AI analysis of the scene (if Groq API is configured)
   - Attached image of the frame with detection
4. **Email Delivery**: Sends via Gmail SMTP

## Features

- **HTML Email**: Rich formatting with detection details
- **Image Attachment**: Includes the frame where person was detected
- **Anti-spam**: 5-minute cooldown between alerts
- **AI Integration**: Includes Groq AI analysis in the email
- **Non-blocking**: Email sending runs in background thread

## API Endpoints

### Check Email Status
```bash
curl http://localhost:5000/api/status
```
Returns email configuration status in the JSON response:
- `email_alerts_enabled`: boolean indicating if email is configured
- `last_email_alert`: timestamp of last alert sent
- `email_cooldown_remaining`: seconds until next alert can be sent

### Test Email Function
```bash
curl http://localhost:5000/api/test_email
```
Sends a test email to verify configuration is working.

## Email Content Example

**Subject**: 🚨 Security Alert - Person Detected - 2025-08-25 17:15:30

**Body**:
- Time: 2025-08-25 17:15:30
- Detection: Person detected in security feed
- Confidence: 0.87
- AI Analysis: A person is visible in the security camera frame...

**Attachment**: alert_20250825_171530.jpg

## Troubleshooting

### Common Issues

1. **"Email configuration incomplete"**
   - Check all three environment variables are set in `.env`
   - Ensure no extra spaces in the values

2. **Authentication failed**
   - Use App Password, not regular Gmail password
   - Verify 2-Step Verification is enabled on Gmail account

3. **SMTP connection errors**
   - Check internet connection
   - Verify Gmail SMTP settings (smtp.gmail.com:587)

4. **No emails being sent during detection**
   - Check cooldown period (5 minutes between alerts)
   - Verify person detection is working in the video stream

### Testing

Run the test endpoint to verify email functionality:
```bash
curl http://localhost:5000/api/test_email
```

Expected response:
```json
{
  "success": true,
  "message": "Test email sent successfully!"
}
```

## Security Considerations

- Store email credentials securely in `.env` file
- Don't commit `.env` file to version control
- Consider using dedicated email account for alerts
- Monitor email usage to avoid hitting Gmail limits

## Configuration Options

You can modify these settings in `web_app.py`:

- `email_alert_cooldown`: Time between alerts (default: 300 seconds)
- SMTP server and port (currently configured for Gmail)
- Email template and content

## Integration with AI Analysis

When both email alerts and Groq AI are configured:
- Person detection triggers both AI analysis and email alert
- Email includes the AI analysis results
- Both features have independent cooldowns for optimal performance
