# Seasaw LB trajectory (public web)

GitHub Actions上でKaggle公式Python APIを実行し、`pokemon-tcg-ai-battle`におけるSeasawの有効な直近2 submissionを取得します。`EpisodeAgents.parquet`からFinal Submission Deadline以降のスコア推移を描画し、GitHub Pagesで公開します。Kaggle API tokenはActions内だけで使い、公開サイトへは出力しません。

Kaggle公式ガイドでも、Simulation Competitionでは`team-submissions`で有効submissionを取得し、`episodes`でsubmissionのepisodeを確認する手順が案内されています。[公式ガイド](https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md)

## GitHub Pagesで公開

1. このディレクトリをGitHubリポジトリのrootへpushします。
2. Repository secret `KAGGLE_API_TOKEN`へ、Seasawチーム所属アカウントのKaggle API tokenを登録します。
3. APIから締切を取得できない場合だけ、Repository variable `FINAL_SUBMISSION_DEADLINE`へ`2026-08-01T00:00:00Z`のようなUTC日時を登録します。
4. **Settings → Pages → Build and deployment → Source** で **GitHub Actions** を選択します。
5. **Actions → Publish Seasaw LB graph → Run workflow** を一度実行します。

以後、Workflowが10分ごとにKaggle APIから再取得してPagesを更新します。ページ自体にも600秒の自動再読み込みを設定しています。

公開URLは通常、`https://<GitHubユーザー名>.github.io/<リポジトリ名>/`です。生成されるページにはPNGグラフとCSVダウンロードが含まれます。

## ローカル確認（任意）

```bash
pip install -r requirements.txt
export KAGGLE_API_TOKEN='...'
python update_site.py
```

既存のparquetだけで描画する場合は、次のコマンドを使えます。

```bash
python plot_lb_trajectory.py --parquet /path/to/EpisodeAgents.parquet
```
