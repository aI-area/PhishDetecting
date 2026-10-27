

# Constants (unchanged)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1 
RANDOM_SEED = 42

SUSPICIOUS_TERMS = [
    "webscr", "cmd", "dispatch", "cgi-bin", "secure-code", "webcss", 
    "limitation", "session-expired", "confirm-paypal", "updates-paypal",
    "customercare", "memberinfo", "webmailcogeco", "signintowebmail",
    "customercaremember", "skytool", "pipind", "alivesellive",
    "secure-login", "verify-account", "confirm-identity", "update-billing",
    "password-reset", "account-verify", "banking-login", "payment-update",
    "identity-confirm", "signin-verify", "password-update", "secure-signin",
    "wallet-restore", "verify-now", "account-suspended", "unusual-activity",
    "account-limitation", "security-alert", "validation-required", "authorize-login",
    "validate-account", "authentication-required", "myaccount-login",
    "000webhostapp", "yolasite",
    "verification", "authenticate", "office365", "paypal",
    "weeblysite", "godaddysites", "wixsite", "000webhost", "netlify", "herokuapp",
    "wordpress", "blogspot", "sites.google", "sharepoint", "s3.amazonaws",
    "drive.google", "docs.google", "dropbox", "mediafire", "onedrive", "box",
    "paypal", "bank", "credit", "debit", "loan", "investment", "transfer", "wire",
    "microsoft", "apple", "google", "facebook", "instagram", "amazon", "netflix",
    "twitter", "linkedin", "adobe", "zoom", "outlook", "office", "icloud",
    "wp-includes", "wp-admin", "wp-content", "plugins", "tinymce", "jscripts",
    "mce", "editors", "tiny_mce", "xmlrpc", "cpanel", "phpmyadmin", "admin",
    "service", "update-payment", "verification", "index.php", "login.php", "signin",
    "validate", "authorize", "authentication", "auth", "session", "websrc", "session",
    "myaccount", "confirm", "verification.php", "verification.html", "update.html",
    ".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ".action", ".do",
    "pdf", "doc", "docx", "xls", "xlsx", "csv", "attachment", "invoice"
]

RARE_TLDS = [
    ".xyz", ".top", ".club", ".info", ".biz", ".cf", ".ml", ".gq", ".tk",
    ".info", ".site", ".online", ".buzz", ".live", ".icu",
    ".io", ".app", ".me", ".co", ".ga", ".cc", ".cfd", ".design", ".shop",
    ".store", ".tech", ".link", ".space", ".fun", ".today", ".live",
    ".click", ".work", ".website", ".world", ".digital", ".network"
]

SUSPICIOUS_DOMAINS = [
    "weebly.com", "webflow.io", "godaddysites.com", "wixsite.com",
    "netlify.app", "herokuapp.com", "000webhost.com", "wordpress.com",
    "blogspot.com", "sites.google.com", "firebaseapp.com", "github.io",
    "crazydomains.com", "squarespace.com", "myshopify.com", "glitch.me",
    "repl.co", "surge.sh", "now.sh", "vercel.app", "pages.dev"
]

MAJOR_LEGITIMATE_SITES = [
    "google.com", "microsoft.com", "apple.com", "amazon.com", "youtube.com",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "github.com",
    "yahoo.com", "netflix.com", "wikipedia.org", "baidu.com", "live.com",
    "bing.com", "ebay.com", "pinterest.com", "reddit.com", "dropbox.com",
    "whatsapp.com", "bitly.com", "bit.ly", "tinyurl.com", "t.co"
]

TRUSTWORTHY_DOMAINS = [
    ".edu", ".gov", ".mil", ".ac.uk", ".edu.au", ".gc.ca", ".gob.mx", ".gov.uk",
    "university", "school", "college", "institute", "academy", "campus"
]
