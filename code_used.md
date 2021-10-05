git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.h5' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Test_raspberry/Selected_Prior2.h5' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.txt' --prune-empty --tag-name-filter cat -- --all






<!-- git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Porto/Setup/Grid/Base.txt' -->
