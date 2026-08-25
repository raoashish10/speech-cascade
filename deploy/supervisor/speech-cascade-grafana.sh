#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Grafana"

mkdir -p /var/log/grafana /run/grafana
cd /usr/share/grafana
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
  cfg:default.security.admin_user=admin \
  cfg:default.security.admin_password="${GRAFANA_ADMIN_PASSWORD:-speechcascade}" 2>&1
