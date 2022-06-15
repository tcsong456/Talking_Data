if [ ! -f talkingdata-mobile-user-demographics.zip ]
then
  echo "downloading data from kaggle"
  kaggle competitions download -c talkingdata-mobile-user-demographics
else
  echo "data already exists"
fi

if [ ! -d data ]
then
  mkdir data
fi
unzip -o talkingdata-mobile-user-demographics.zip -d data
rm talkingdata-mobile-user-demographics.zip

data_files=$(ls data)
for file in $data_files;do
  unzip -o "data/$file" -d data
  rm "data/$file"
done
