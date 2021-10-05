git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch Porto/Delft3D/fig/windconditions.pdf' \
  --prune-empty --tag-name-filter cat -- --all

git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Raspberry/samples_2020.05.01.nc'
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.txt'
=======
>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd
=======
>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd
=======
>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd

git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Porto/Setup/Grid/Base.txt'