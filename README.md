# chatgpt-webapp
chatgptAPIを流用したアプリケーションの作成　

以下のサイトのコードをコピーして作成
chatgptAPIとLangChainを利用したwebサービスの作成を学ぶことができた


# 作成メモ


openAPIのAPIキーを設定する必要がある
また利用するにはクレジットカードを登録する必要がある

[ChatGPTに聞きながらChatGPT APIの使用手順を書いてみた - Qiita](https://qiita.com/LingmuSajun/items/8ac6b016e0ecc864851e)
[OpenAIでRateLimitErrorが出たときの対応方法 日本語訳](https://zenn.dev/masaki_mori72/scraps/3ad2a70353e9b8)

```jsx
export OPENAI_API_KEY="APIキー"
```

**最重要パラメーター `temperature`**
AIによる解答をどのくらい関連性高く回答させるか、１だとランダム性が高い回答、０だと確実性、多様性が低い回答になる

requirements.txtにライブラリバージョンを記載しておく
[App dependencies - Streamlit Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

# 資料
[つくりながら学ぶ！AIアプリ開発入門 - LangChain & Streamlit による ChatGPT API 徹底活用](https://zenn.dev/ml_bear/books/d1f060a3f166a5)
[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
