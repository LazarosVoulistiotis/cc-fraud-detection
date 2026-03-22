"""logging_config.py

Structured JSON logging for the FastAPI inference service.
Logs are emitted to stdout in machine-readable format.
"""

import logging
import sys # τα logs θα γράφονται στην standard output (sys.stdout)

import structlog # προσθέτει structured/event-based λογική


# Η βασική function που ρυθμίζει το logging σύστημα. Ο σκοπός της δεν είναι να επιστρέψει object, αλλά να κάνει global configuration
def configure_logging() -> None:
    """
    Configure stdlib logging + structlog for JSON output.
    Safe to call once at application startup.
    """
    # Φτιαχνουμε processor που προσθέτει timestamp σε κάθε log event. Ένας processor είναι μια function/μονάδα που παίρνει ένα log event και το εμπλουτίζει ή το μετασχηματίζει
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True) # timestamp να είναι σε ISO format π.χ. 2026-03-20T14:25:10.123456Z

    shared_processors = [
        structlog.contextvars.merge_contextvars, # Αυτός ο processor ενσωματώνει context variables μέσα στο log event. contextvars = ένας τρόπος να κρατάς request-scoped context
        timestamper,
        structlog.stdlib.add_log_level, # Αυτός ο processor προσθέτει το log level στο event dict.
    ]

    logging.basicConfig(
        format="%(message)s", # να τυπώνεται μόνο το message επειδή το structlog θα έχει ήδη δημιουργήσει το πλήρες JSON log ως message, δεν θες το stdlib logging να τυπώνει επιπλέον prefixes και να “χαλάει” το JSON
        stream=sys.stdout,
        level=logging.INFO, # minimum severity level
    )

    structlog.configure(
        processors=[
            *shared_processors, # Το * κάνει unpack τη λίστα shared_processors.
            structlog.processors.dict_tracebacks, # Αυτός ο processor μετατρέπει exceptions / tracebacks σε structured dictionary μορφή.
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger, # ο BoundLogger υποστηρίζει structured contextual logging
        logger_factory=structlog.stdlib.LoggerFactory(), # Η LoggerFactory() εδώ λέει στο structlog να χρησιμοποιεί stdlib loggers από το Python logging.
        cache_logger_on_first_use=True, # Αυτό λέει στο structlog να κάνει cache τον logger αφού χρησιμοποιηθεί πρώτη φορά.
    )

# Αυτή είναι μια helper function που επιστρέφει logger instance.
def get_logger():
    return structlog.get_logger("fraud_api")