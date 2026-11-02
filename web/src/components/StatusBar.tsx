// iOS-style status bar shown only on the desktop device mock-up (see styles.css).
export default function StatusBar() {
  return (
    <div className="status-bar" aria-hidden="true">
      <span className="status-time">9:41</span>
      <span className="dynamic-island" />
      <div className="status-icons">
        <svg width="18" height="12" viewBox="0 0 18 12" fill="#000">
          <rect x="0" y="8" width="3" height="4" rx="1" />
          <rect x="5" y="5.5" width="3" height="6.5" rx="1" />
          <rect x="10" y="3" width="3" height="9" rx="1" />
          <rect x="15" y="0" width="3" height="12" rx="1" />
        </svg>
        <svg width="16" height="12" viewBox="0 0 16 12" fill="none">
          <path
            d="M1 4.2C4.9 0.7 11.1 0.7 15 4.2"
            stroke="#000"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
          <path
            d="M3.9 7.3C6.5 5 9.5 5 12.1 7.3"
            stroke="#000"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
          <circle cx="8" cy="10.4" r="1.4" fill="#000" />
        </svg>
        <svg width="28" height="14" viewBox="0 0 28 14" fill="none">
          <rect
            x="0.75"
            y="0.75"
            width="23.5"
            height="12.5"
            rx="3.5"
            stroke="#000"
            strokeWidth="1.5"
          />
          <rect x="2.5" y="2.5" width="17" height="9" rx="2" fill="#000" />
          <rect x="25.5" y="4.5" width="2" height="5" rx="1" fill="#000" />
        </svg>
      </div>
    </div>
  );
}
