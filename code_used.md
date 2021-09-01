git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch Porto/Delft3D/fig/windconditions.pdf' \
  --prune-empty --tag-name-filter cat -- --all

git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Raspberry/samples_2020.05.01.nc'

git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Porto/Setup/Grid/Base.txt'