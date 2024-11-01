from binoculars import Binoculars

bino = Binoculars()

# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"
sample_string = '''春の陽気が街を包む季節となりました。桜の花びらが風に舞い、人々の心を和ませています。公園では子供たちが元気に遊び、お年寄りがベンチでゆっくりと過ごしています。カフェのテラス席では、若いカップルがコーヒーを楽しみながら談笑しています。
通りを歩けば、新しい生活を始める人々の姿が目に付きます。新入社員や新入生たちが、期待と不安が入り混じった表情で歩いています。花屋の店先には色とりどりの花が並び、行き交う人々の目を楽しませています。
この季節は新しい出会いと別れの時期でもあります。友人との別れを惜しみつつ、新たな環境での出会いを心待ちにしている人も多いでしょう。春の訪れとともに、人々の心にも新たな希望の芽が芽生えているようです。'''

print(bino.compute_score(sample_string))  # 0.75661373
print(bino.predict(sample_string))  # 'Most likely AI-Generated'
