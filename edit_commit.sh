#!/bin/bash

# Create a temporary script to edit the commit message
cat > /tmp/edit_commit.sh << 'EOF'
#!/bin/bash
# Replace the commit message in the rebase file
sed -i 's/E-commerce Analytics Platform for Vercel deployment/E-commerce Analytics Dashboard Platform/g' "$1"
EOF

chmod +x /tmp/edit_commit.sh

# Set the editor
export EDITOR="/tmp/edit_commit.sh"

# Perform interactive rebase
git rebase -i fde8a70~1

echo "Commit message updated!"
