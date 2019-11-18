if [ ! -d datasets/linemod/Linemod_preprocessed ];then
echo 'Downloading the preprocessed LineMOD dataset'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YFUra533pxS_IHsb9tB87lLoxbcHYXt8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YFUra533pxS_IHsb9tB87lLoxbcHYXt8" -O Linemod_preprocessed.zip && rm -rf /tmp/cookies.txt \
&& unzip Linemod_preprocessed.zip \
&& mv Linemod_preprocessed/ datasets/linemod/ \
&& rm Linemod_preprocessed.zip
fi

echo 'Download Complete'
