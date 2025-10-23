#!/bin/bash

# Create a backup branch
git branch backup-before-rename

# Reset to the commit before the one we want to rename
git reset --hard 07844c7

# Cherry-pick the commits we want to keep, but with new messages
git cherry-pick 894a34c
git commit --amend -m "Add lightweight HTML dashboard for Vercel deployment"

git cherry-pick ecf0af3
git commit --amend -m "Update E-commerce Analytics Dashboard Platform for Vercel deployment"

git cherry-pick fde8a70
git commit --amend -m "E-commerce Analytics Dashboard Platform"

# Cherry-pick the remaining commits
git cherry-pick 690cf18
git cherry-pick 79a5e98
git cherry-pick d06e883
git cherry-pick c004138
git cherry-pick fe3a4d4
git cherry-pick e78a3c4
git cherry-pick aac8bbf
git cherry-pick c4532f1

echo "Commit messages renamed successfully!"
echo "Backup branch created as 'backup-before-rename'"
