from semantic_router import Route,SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    name= "sentence-transformers/paraphrase-MiniLM-L6-v2",
)
faq = Route(
        name= 'faq',
        utterances= ['What is the return policy of the products?',
                  'Do I get discount with the HDFC credit card?',
                  'How can I track my order?',
                  'What payment methods are accepted?',
                  'How long does it take to process a refund?',
                  'What should I do if I receive a damaged product?'],
        score_threshold=0.3
    )

sql = Route(
        name='sql',
        utterances=['I want to buy Nike shoes that have 50% discount',
                'Are there any shoes under Rs. 3000?',
                'Do you have formal shoes in size 9?',
                'Are there any puma shoes on sale',
                'What is the price of puma running shoes?',
                'give me the top 3 shoes in the descending order of rating?'
                ],
        score_threshold=0.4
)

router = SemanticRouter(routes=[faq, sql],encoder=encoder,auto_sync="local")

if __name__=='__main__':
    print(router("show me top 3 nike shoes with the rating higher than 4.5").name)
    print(router("Pink puma shoes in price range 5000 to 1000").name)




