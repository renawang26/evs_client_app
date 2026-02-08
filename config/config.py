"""
Configuration settings for the application
"""

# User authentication settings
AUTH_CONFIG = {
    # Whether to bypass login completely (set to True to disable authentication)
    'BYPASS_LOGIN': True,

    # Default user for bypass login (will be used when BYPASS_LOGIN is True)
    'DEFAULT_USER': {
        'email': 'admin@example.com',
        'is_admin': True
    },

    # Whether to enable email verification
    'ENABLE_EMAIL_VERIFICATION': False,

    # Whether to enable user registration
    'ENABLE_REGISTRATION': False,

    # Email validation settings
    'EMAIL_VALIDATION': {
        'ENABLE': True,  # Whether to validate email format
        'PATTERN': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'  # Email regex pattern
    },

    # SMTP settings (only used if email verification is enabled)
    'SMTP': {
        'SERVER': 'smtp.gmail.com',
        'PORT': 587,
        'USERNAME': 'your.email@gmail.com',
        'PASSWORD': 'your-app-password',
        'FROM_EMAIL': 'your.email@gmail.com',
        # Email queue settings
        'MAX_EMAILS_PER_MINUTE': 20,  # Rate limit for emails per minute
        'MAX_RETRY_ATTEMPTS': 3,      # Maximum retry attempts for failed emails
        'RETRY_DELAY': 300,           # Delay between retries in seconds (5 minutes)
        'QUEUE_ENABLED': True         # Whether to use email queue
    },

    # Password settings
    'PASSWORD': {
        'MIN_LENGTH': 6,
        'REQUIRE_SPECIAL_CHAR': False,
        'REQUIRE_NUMBER': False
    }
}

# Google Cloud settings
GOOGLE_CLOUD_CONFIG = {
    # Directory to store Google Cloud credentials
    'CREDENTIALS_DIR': './keys',
    'CREDENTIALS_FILE': 'google_credentials.json',

    # Google Cloud Speech API settings
    'SPEECH_API': {
        'LANGUAGE_CODE': {
            'en': 'en-US',
            'zh': 'zh-CN'
        },
        'ENCODING': 'LINEAR16',
        'SAMPLE_RATE': 16000
    }
}

# Application base URL
BASE_URL = "http://localhost:8501"