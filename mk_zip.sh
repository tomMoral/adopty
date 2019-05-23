BAD_WORDS="thomas\\|moreau\\|ablin\\|pierre\\|alexandre\\|gramfort\\|inria"

unsafe=$(git grep -i -- "$BAD_WORDS" ":(exclude)mk_zip.sh")
if [[ ! -z $unsafe ]]; then
	echo -e '\033[1;31mThe repository is not anonymous!\n\n\033[0m'
	echo -e 'Found:\n'$unsafe
else
	echo -e '\033[1;32mThe repository seems anonymous.\0033[0m\n\n'
fi



# Check that all included examples are described in the README.md
for f in examples/*.py; do
	is_in_readme=$(grep $f README.md)
	if [[ ! -z "$is_in_readme" ]]; then
		echo -e $f':\033[1;32m ok\033[0m'
	else 
		echo -e '\033[1;31mRemove '$f'\033[0m'
	fi
done

# Get all files in the repository and avoid includingn git files
echo -e '\n======  \033[1;33mCreate the archive\033[0m =======\n'
list_files=$(git ls-tree -r --name-only anonymous | grep -v ".git\\|mk_zip" )
zip adopty.zip $list_files

# Display the contents of the zip
echo -e '\n======  \033[1;33mCreate the archive\033[0m =======\n'
unzip -l adopty.zip
