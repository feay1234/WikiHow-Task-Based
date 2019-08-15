input=$1
while IFS= read -r line
do
  python gquestions.py query "$line" en --cs
done < "$input"
