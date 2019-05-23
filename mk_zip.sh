BAD_WORDS="thomas\\|moreau\\|ablin\\|pierre\\|alexandre\\|gramfort\\|inria"
BASENAME="learned_step_sizes"
FINAL_PDF="${BASENAME}.pdf"
FULL_PDF="${BASENAME}_with_supplementary.pdf"
ZIP_NAME="${BASENAME}_with_supplementary_and_code.zip"


if [[ ! -z "$(pdffonts $FULL_PDF | grep TrueType)" ]]; then
	echo -e '\033[1;31mBad font in the generated pdf file\033[0m'
	exit
else
	echo -e '\033[1;32mNo TrueType font in pdf\033[0m'
fi


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

echo -e '\n======  \033[1;33mCreate the archive\033[0m =======\n'

# If adopty.zip already exists, remove it first to avoid leaking files.
if [[ -f $ZIP_NAME ]]; then
	echo "Remove previous archive"
    rm $ZIP_NAME
fi
echo -e 'Create archive \033[1;32m'$ZIP_NAME'\033[0m'
# Get all files in the repository and avoid includingn git files
list_files=$(git ls-tree -r --name-only anonymous | grep -v ".git\\|mk_zip" )
zip -q $ZIP_NAME $list_files $FULL_PDF


echo -e '\n======  \033[1;33mCrop pdf for final submission\033[0m =======\n'
# Generate the submission file
pdftk $FULL_PDF cat 1-10 output $FINAL_PDF
echo -e 'Submission file: \033[1;32m'$FINAL_PDF'\033[0m'
