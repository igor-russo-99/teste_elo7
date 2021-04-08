import argparse as ap
import joblib
import pandas as pd
import ast
import string

with open('data/stopwords_portuguese', 'r') as reader:
   stopwords = [line.strip() for line in reader]


def text_preprocess(text):
    """
    1. remove pontuação
    2. remove stop words
    3. retorna lista de palavras resultante
    """
    
    sem_pontuacao = [char for char in text if char not in string.punctuation]
    
    sem_pontuacao = ''.join(sem_pontuacao)
    
    sem_stop_words = [word for word in sem_pontuacao.split() if word.lower() not in stopwords]
    
    return sem_stop_words


def category(features_dict):

    classificador_categoria_produto = joblib.load('models/classificador_categoria_produto.pkl')

    query = features_dict['query']
    price = features_dict['price']

    consulta = pd.DataFrame(data={'query':[query], 'price': [price]})
    pred = classificador_categoria_produto.predict(consulta.values)[0]
    
    return pred   


def intent(query):
 
    classificador_intencao   = joblib.load('models/classificador_intencao.pkl')
    classes_intencao_usuario = joblib.load('data/classes_intencao_usuario.pkl')

    df_query = pd.DataFrame(data={'query':[query]})
    classe_id = int(classificador_intencao.predict(df_query.values))

    return classes_intencao_usuario[classe_id]


def recomenda_produtos_mais_desejados(categoria, classe_intencao):

    df_produtos = pd.read_csv('data/produtos.csv')

    df_filtrado = df_produtos[(df_produtos['category'] == categoria) & (df_produtos['price'] >= classe_intencao['price_min']) & (df_produtos['price'] <= classe_intencao['price_max'])]

    produtos_recomendados = df_filtrado.sort_values(by=['view_counts'], ascending=False).head(10)

    return produtos_recomendados[['product_id', 'title']]
    

def recommendation(query):

    classe_intencao = intent(query)
    price_median = classe_intencao['price_median']
    categoria  = category({'query':query, 'price':price_median})

    produtos_recomendados = recomenda_produtos_mais_desejados(categoria, classe_intencao)

    return {'classe': classe_intencao, 'categoria': categoria, 'produtos': produtos_recomendados.to_dict('list')}    

if __name__ == '__main__':

    parser = ap.ArgumentParser(description="Teste DS")
    parser.add_argument("--category")    
    parser.add_argument("--intent")
    parser.add_argument("--recommendation")
    args, leftovers = parser.parse_known_args()
    
    if args.category is not None:
        features = ast.literal_eval(args.category)
        cat = category(features)
        print(f'"{cat}"')

    elif args.intent is not None:
        classe_intencao = intent(args.intent)
        print(f'"{classe_intencao["nome"]}"')

    elif args.recommendation is not None:
        recomendacao = recommendation(args.recommendation)

        print(f'"{recomendacao["categoria"]}"')
        print(f'"{recomendacao["classe"]["nome"]}"')

        produtos = pd.DataFrame(recomendacao["produtos"])

        for product_id, title in zip(produtos['product_id'].to_list(), produtos['title'].to_list()):
            print(f'"{product_id},{title}"')       

    else:
        parser.print_help()      
