git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.h5' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Test_raspberry/Selected_Prior2.h5' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard/Sigma_sal.txt' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/samples_2020.05.01.nc' --prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Nidelva/GIS/Norway_shapefile/no_1km.shp' --prune-empty --tag-name-filter cat -- --all

git pull origin master --allow-unrelated-histories

git rm --cached -r **/.idea
git rm --cached -r **/__pycache__
git rm --cached -r *.DS_Store

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch Adaptive_script/Porto/Onboard.zip' --prune-empty --tag-name-filter cat -- --all

scp -r target user@192.168.1.100:"/file\\ path\\ with\\ spaces/myfile.txt"
- black space is replaced by \\
<!-- git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch Porto/Setup/Grid/Base.txt' -->
