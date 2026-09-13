#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Grafana"

mkdir -p /var/log/grafana /run/grafana
cd /usr/share/grafana

# Grafana's login POST enforces a same-origin check against its own
# server.root_url -- reachable externally only through Caddy on
# PUBLIC_IPADDR:VAST_TCP_PORT_10200 (a different host:port than the
# internal 127.0.0.1:13000 it binds to), so without this it always
# rejects the login form with "Login failed / origin not allowed" no
# matter how the request actually got there. Built dynamically since the
# public IP/port are assigned per-instance.
GRAFANA_ROOT_URL="http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_10200}/"
# root_url alone fixes the /login page's own origin check, but NOT the
# separate origin check Grafana's API layer applies to POST endpoints
# like /api/ds/query (the one every panel uses to fetch data) -- that one
# is gated by [security] csrf_trusted_origins specifically, a distinct
# setting from root_url, and defaults to empty. Without it, every panel's
# query comes back "origin not allowed" (visible directly in the response
# body) even with a perfectly valid auth cookie attached -- confirmed via
# a real browser's Network tab: correct cookie present, request still
# 403's on this exact text. No trailing slash here, matching the bare
# scheme+host+port an Origin header actually contains.
GRAFANA_TRUSTED_ORIGIN="http://${PUBLIC_IPADDR}:${VAST_TCP_PORT_10200}"

pty /usr/share/grafana/bin/grafana server \
  --config=/etc/grafana/grafana.ini \
  --homepath=/usr/share/grafana \
  --pidfile=/run/grafana/grafana-server.pid \
  --packaging=deb \
  cfg:default.paths.logs=/var/log/grafana \
  cfg:default.paths.data=/var/lib/grafana \
  cfg:default.paths.plugins=/var/lib/grafana/plugins \
  cfg:default.paths.provisioning=/workspace/speech-cascade-inference/monitoring/grafana/provisioning \
  cfg:default.server.http_addr=127.0.0.1 \
  cfg:default.server.http_port=13000 \
  cfg:default.server.root_url="${GRAFANA_ROOT_URL}" \
  cfg:default.security.admin_user=admin \
  cfg:default.security.admin_password="${GRAFANA_ADMIN_PASSWORD:-speechcascade}" \
  cfg:default.security.csrf_trusted_origins="${GRAFANA_TRUSTED_ORIGIN}" \
  cfg:default.auth.anonymous.enabled=true \
  cfg:default.auth.anonymous.org_role=Viewer 2>&1
