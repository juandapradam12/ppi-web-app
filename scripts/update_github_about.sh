#!/usr/bin/env bash
# Update GitHub repo name, About description, and topics.
# Run this on your machine (owner account) — Cloud Agent tokens cannot PATCH repo settings.
#
#   bash scripts/update_github_about.sh
#
set -euo pipefail

OWNER="${GITHUB_OWNER:-juandapradam12}"
OLD_NAME="${GITHUB_OLD_REPO:-ppi-web-app}"
NEW_NAME="${GITHUB_NEW_REPO:-PlayerPerformanceIndex}"

# GitHub About has a ~350 character soft limit for best display.
DESCRIPTION='Turn match stats into clear, role-aware player rankings. Expert weights define the score; transparent ML approximates it live — with attribution, uncertainty, and El Clásico demos. Built for scouts, analysts, and data portfolios that need explainable sports AI.'

TOPICS=(
  soccer
  football-analytics
  sports-analytics
  player-performance
  machine-learning
  interpretable-ml
  scikit-learn
  streamlit
  data-science
  python
  feature-attribution
  ranking
  barcelona
  fbref
  portfolio-project
)

echo "→ Updating description & homepage on ${OWNER}/${OLD_NAME}"
gh api -X PATCH "repos/${OWNER}/${OLD_NAME}" \
  -f description="$DESCRIPTION" \
  -f homepage="https://github.com/${OWNER}/${NEW_NAME}" \
  -F has_issues=true \
  -F has_projects=false \
  -F has_wiki=false \
  >/dev/null

echo "→ Setting topics"
# Topics API expects JSON array
topics_json=$(printf '%s\n' "${TOPICS[@]}" | jq -R . | jq -s .)
gh api -X PUT "repos/${OWNER}/${OLD_NAME}/topics" \
  -H "Accept: application/vnd.github.mercy-preview+json" \
  --input - <<<"{\"names\": $topics_json}" \
  >/dev/null

echo "→ Renaming repository to ${NEW_NAME}"
gh api -X PATCH "repos/${OWNER}/${OLD_NAME}" \
  -f name="$NEW_NAME" \
  >/dev/null

echo
echo "Done."
echo "  New URL: https://github.com/${OWNER}/${NEW_NAME}"
echo "  Description: $DESCRIPTION"
echo "  Topics: ${TOPICS[*]}"
echo
echo "Optional: update local remotes:"
echo "  git remote set-url origin git@github.com:${OWNER}/${NEW_NAME}.git"
