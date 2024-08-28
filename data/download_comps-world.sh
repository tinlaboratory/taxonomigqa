DIR=data/comps-world
mkdir -p $DIR

wget https://raw.githubusercontent.com/kanishkamisra/world/main/data/concept_matrix.txt -P $DIR/

wget https://raw.githubusercontent.com/kanishkamisra/world/main/data/concept_senses.csv -P $DIR/

wget https://raw.githubusercontent.com/kanishkamisra/world/main/data/categories.csv -P $DIR/

wget https://raw.githubusercontent.com/kanishkamisra/world/main/data/feature_lexicon.csv -P $DIR/

wget https://raw.githubusercontent.com/kanishkamisra/world/main/data/feature_metadata.csv -P $DIR/

wget https://raw.githubusercontent.com/kanishkamisra/world/main/data/xcslb_compressed.csv -P $DIR/