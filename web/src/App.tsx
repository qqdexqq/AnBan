import AssistantSheet from "./components/AssistantSheet";
import Compliance from "./components/Compliance";
import Home from "./components/Home";
import JuryPanel from "./components/JuryPanel";
import Notifications from "./components/Notifications";
import SettingsScreen from "./components/Settings";
import StatusBar from "./components/StatusBar";
import { useApp } from "./state";

export default function App() {
  const { ready, screen, assistantOpen, juryPanel } = useApp();

  return (
    <div className="stage">
      <div className="phone">
        <StatusBar />
        {!ready ? (
          <div className="screen" style={{ alignItems: "center", justifyContent: "center" }}>
            <span className="muted">AnBan 安办…</span>
          </div>
        ) : (
          <>
            {screen === "home" && <Home />}
            {screen === "settings" && <SettingsScreen />}
            {screen === "compliance" && <Compliance />}
            {assistantOpen && <AssistantSheet />}
          </>
        )}
        <Notifications />
        <div className="home-indicator" />
      </div>
      {ready && juryPanel && <JuryPanel />}
    </div>
  );
}
