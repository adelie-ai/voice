-- Adele AEC stream router — WirePlumber `select-target` hook (Linux only).
--
-- Pins Adele's own audio streams to the echo-cancelled virtual nodes created
-- by aec/pipewire/99-adele-echo-cancel.conf, so her TTS is removed from the mic
-- and she stops hearing / interrupting / answering herself. Scoped to Adele:
-- matches the daemon and the embedding clients (adele-voice / adele-gtk /
-- adele-tui) by their PipeWire stream identity; every other app and the system
-- default routing are left untouched.
--
-- Why a hook and not a static rule: in WirePlumber 0.5 neither `stream.rules`
-- nor `node.rules` can retarget an arbitrary *client* stream, and setting the
-- target via metadata only applies under `linking.allow-moving-streams`. The
-- supported path is to inject the target at link time on the `select-target`
-- event — this runs before `find-defined-target`, isn't a "move", and is
-- exactly what the shipped find-user-target.lua.example demonstrates.
--
-- Installed to ~/.config/wireplumber/scripts/ by `just install-aec`, and loaded
-- by the companion 51-adele-aec.conf component declaration.

lutils = require ("linking-utils")
log = Log.open_topic ("s-linking")

-- True when this stream belongs to Adele (the daemon or an embedding client).
-- Matches on any identity prop carrying the "adele" binary stem, so it covers
-- adele-voice / adele-gtk / adele-tui without enumerating each.
local function is_adele (props)
  for _, key in ipairs ({ "application.process.binary", "node.name", "application.name" }) do
    local v = props [key]
    if v and string.find (string.lower (v), "adele", 1, true) then
      return true
    end
  end
  return false
end

SimpleEventHook {
  name = "linking/adele-aec-target",
  before = "linking/find-defined-target",
  interests = {
    EventInterest {
      Constraint { "event.type", "=", "select-target" },
    },
  },
  execute = function (event)
    local source, om, si, si_props, si_flags, target =
        lutils:unwrap_select_target_event (event)

    -- Respect an explicit user pick; ignore everything that isn't Adele.
    if target or not is_adele (si_props) then
      return
    end

    local media_class = si_props ["media.class"] or ""
    local want
    if media_class == "Stream/Input/Audio" then
      want = "adele_ec_source"
    elseif media_class == "Stream/Output/Audio" then
      want = "adele_ec_sink"
    else
      return
    end

    -- Degrade, don't break: if the echo-cancel node isn't loaded (module
    -- absent, or still starting up), leave the target unset so the normal hooks
    -- pick a default. Adele then runs WITHOUT cancellation rather than failing
    -- to link at all.
    local node = om:lookup { Constraint { "node.name", "=", want } }
    if not node then
      log:info (si, string.format ("adele-aec: %s not present; leaving default target", want))
      return
    end

    log:info (si, string.format ("adele-aec: routing %s to %s", media_class, want))
    event:set_data ("target", node)
  end
}:register ()
