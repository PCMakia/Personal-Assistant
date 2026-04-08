from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class OutlookAppointmentResult:
    entry_id: str | None
    subject: str
    start: datetime
    duration_minutes: int


class OutlookCalendarError(RuntimeError):
    pass


def _require_windows() -> None:
    if not sys.platform.startswith("win"):
        raise OutlookCalendarError("Outlook calendar automation is only supported on Windows.")


def create_outlook_appointment(
    *,
    subject: str,
    body: str = "",
    start: datetime,
    duration_minutes: int = 60,
    reminder_minutes_before_start: int = 15,
) -> OutlookAppointmentResult:
    """Create and save an Outlook Calendar appointment via COM automation.

    Permission/session caveats (common failure modes):
    - Outlook must be installed, and the current user session must allow COM automation.
    - Some corporate policies block programmatic access; catch exceptions and show guidance.
    """
    _require_windows()

    subj = (subject or "").strip()
    if not subj:
        raise OutlookCalendarError("Missing required field: subject")
    if not isinstance(start, datetime):
        raise OutlookCalendarError("Missing/invalid required field: start datetime")

    mins = int(duration_minutes)
    mins = max(1, mins)
    rem = int(reminder_minutes_before_start)
    rem = max(0, rem)

    try:
        # COM initialization is required when called from worker threads.
        import pythoncom  # type: ignore

        pythoncom.CoInitialize()
        try:
            import win32com.client  # type: ignore

            outlook = win32com.client.Dispatch("Outlook.Application")
            appointment = outlook.CreateItem(1)  # 1 = AppointmentItem
            appointment.Subject = subj
            appointment.Body = (body or "").strip()
            appointment.Start = start
            appointment.Duration = mins
            appointment.ReminderSet = True
            appointment.ReminderMinutesBeforeStart = rem
            appointment.Save()
            entry_id = None
            try:
                entry_id = str(getattr(appointment, "EntryID", "") or "") or None
            except Exception:
                entry_id = None
            return OutlookAppointmentResult(
                entry_id=entry_id,
                subject=subj,
                start=start,
                duration_minutes=mins,
            )
        finally:
            pythoncom.CoUninitialize()
    except OutlookCalendarError:
        raise
    except Exception as exc:
        raise OutlookCalendarError(
            "Failed to create Outlook calendar event. "
            "Check that Outlook is installed and available in the current user session, "
            "and that programmatic access is allowed by your Office security settings/policies. "
            f"Underlying error: {exc}"
        ) from exc

