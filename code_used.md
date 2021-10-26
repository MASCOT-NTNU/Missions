git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.h5' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Test_raspberry/Selected_Prior2.h5' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.txt' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/samples_2020.05.01.nc' --prune-empty --tag-name-filter cat -- --all

git rm --cached -r **/.idea
git rm --cached -r **/__pycache__
git rm --cached -r *.DS_Store

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard.zip' --prune-empty --tag-name-filter cat -- --all



<!-- git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Porto/Setup/Grid/Base.txt' -->
