# Standard modules
import os
import sys
from typing import Union

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro' # 日本語をプロット内部に記述するため
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
import ginza

class GiNZANaturalLanguageProcessing(object):
  def __init__(self, model: str='ja_ginza_electra', split_mode: str='C'):
    self.nlp = spacy.load(model)
    ginza.set_split_mode(self.nlp, split_mode)

  # 文境界解析
  def get_sentences(self, text: str) -> list[str]:
    doc = self.nlp(text)
    return doc.sents

  # 文節
  def get_bunsetu_spans(self, text: str) -> list[str]:
    doc = self.nlp(text)
    bunsetu = ginza.bunsetu_spans(doc)
    return bunsetu

  def get_bunsetu_phrase_spans(self, text: str) -> list[str]:
    doc = self.nlp(text)
    bunsetu_phrase = ginza.bunsetu_phrase_spans(doc)
    return bunsetu

  def get_bunsetu_syntaxes(self, text: str) -> list[tuple]:
    doc = self.nlp(text)
    dependencies = []
    for sent in doc.sents:
      for span in ginza.bunsetu_spans(sent):
        for token in span.lefts:
          dependencies.append((token, span))
    return dependencies

  # 形態素解析
  def print_token_syntaxes(self, text: str) -> None:
    '''
    https://qiita.com/kei_0324/items/400f639b2f185b39a0cf
    https://spacy.io/api/token
    * token.i: トークン番号
    * token.orth_: オリジナルテキスト
    * token._.reading: 読み仮名
    * token.pos_: 品詞(UID)
    *   ---------------------------------------------------------------------------------------------------------------
    *   | UID   | 日本語名　　　　| 説明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 例　　　　　　　　　　　　　|
    *   ---------------------------------------------------------------------------------------------------------------
    *   | ADJ   | 形容詞　　　　　| 名詞や代名詞を修飾する　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 暑い、親切な　　　　　　　　|
    *   | ADP   | 接(前/後)置詞　| 名詞句と結びつき、文中のほかの要素との関連を示す　　　　　　　　　　　　　　　　　　| 〜が、〜へ　　　　　　　　　|
    *   | ADV   | 副詞　　　　　　| 動詞、形容詞、ほかの副詞や分全体を修飾する　　　　　　　　　　　　　　　　　　　　　| すっかり、ずっと　　　　　　|
    *   | AUX   | 助動詞　　　　　| 主語や動詞などと一緒に使われ、動詞だけでは表現できない文の意味や時制などを表現する　| れる、らしい　　　　　　　　|
    *   | CCONJ | 接続詞　　　　　| 分の構成要素同士の関係を示す　　　　　　　　　　　　　　　　　　　　　　　　　　　　| また、そして、しかし　　　　|
    *   | DET   | 限定詞　　　　　| 名詞をより明確に示す　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 一つの　　　　　　　　　　　|
    *   | INTJ  | 感嘆詞　　　　　| 「！」　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | NOUN  | 名詞　　　　　　| 物体、物質、人名、場所など　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 水、犬、東京　　　　　　　　|
    *   | NUM   | 数詞　　　　　　| ０、１０００　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | PART  | 助詞　　　　　　| 言葉に意味を肉付けする　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 〜を、〜が　　　　　　　　　|
    *   | PPON  | 固有名詞　　　　| 一つの特定の単勝を指示する　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| モト冬樹　　　　　　　　　　|
    *   | PROPN | 代名詞　　　　　| 名詞または名詞句の代わりに用いられる　　　　　　　　　　　　　　　　　　　　　　　　| 私、これ、そこ　　　　　　　|
    *   | PUNCT | 句読点　　　　　| 「。」「、」　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | SCONJ | 従属接続詞　　　| 主節の補足説明をする　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| なぜなら、もし　　　　　　　|
    *   | SYM   | 記号　　　　　　| 「？」　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | VERB  | 動詞　　　　　　| 物事の動作や作用、状態、存在などを示す　　　　　　　　　　　　　　　　　　　　　　　| 動く、食べる、咲く　　　　　|
    *   | X     | その他　　　　　| 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   ---------------------------------------------------------------------------------------------------------------
    * token.tag_: 品詞(日本語)
    * token.lemma_: 基本形（名寄せ後)
    * token._.inf: 活用情報
    * token.rank: 頻度のように扱えるかも?
    * token.norm_: 原型
    * token.is_oov: 登録されていない単語か?
    * token.is_stop: ストップワードか?
    * token.has_vector: word2vecの情報を持っているか?
    * token.children: 関連語
    * token.lefts: 関連語(左)
    * token.rights: 関連語(右)
    * token.n_lefts: 関連語(左)の数
    * token.n_rights: 関連語(右)の数
    * token.dep_: 係受けの関連性
    *   ----------------------------------------
    *   | tag        | 意味　　　　　　　　　　　　　　|
    *   ----------------------------------------
    *   | acl        | 名詞節修飾語　　　　　　　　　　|
    *   | advcl      | 副詞節修飾語　　　　　　　　　　|
    *   | advmod     | 副詞修飾語　　　　　　　　　　　|
    *   | amod       | 形容詞修飾語　　　　　　　　　　|
    *   | appos      | 同格　　　　　　　　　　　　　　|
    *   | aux        | 助動詞　　　　　　　　　　　　　|
    *   | case       | 格表示　　　　　　　　　　　　　|
    *   | cc         | 等位接続詞　　　　　　　　　　　|
    *   | ccomp      | 捕文　　　　　　　　　　　　　　|
    *   | clf        | 類別詞　　　　　　　　　　　　　|
    *   | compound   | 複合名詞　　　　　　　　　　　　|
    *   | conj       | 結合詞　　　　　　　　　　　　　|
    *   | cop        | 連結詞　　　　　　　　　　　　　|
    *   | csubj      | 主部　　　　　　　　　　　　　　|
    *   | dep        | 不明な依存関係　　　　　　　　　|
    *   | det        | 限定詞　　　　　　　　　　　　　|
    *   | discourse  | 談話要素　　　　　　　　　　　　|
    *   | dislocated | 転置　　　　　　　　　　　　　　|
    *   | expl       | 嘘辞　　　　　　　　　　　　　　|
    *   | fixed      | 固定複数単語表現　　　　　　　　|
    *   | flat       | 同格複数単語表現　　　　　　　　|
    *   | goeswith   | 一単語分割表現　　　　　　　　　|
    *   | iobj       | 間接目的語　　　　　　　　　　　|
    *   | list       | リスト表現　　　　　　　　　　　|
    *   | mark       | 接続詞　　　　　　　　　　　　　|
    *   | nmod       | 名詞修飾語　　　　　　　　　　　|
    *   | nsubj      | 主語名詞　　　　　　　　　　　　|
    *   | nummod     | 数詞修飾語　　　　　　　　　　　|
    *   | obj        | 目的語　　　　　　　　　　　　　|
    *   | obl        | 斜格名詞　　　　　　　　　　　　|
    *   | orphan     | 独立関係　　　　　　　　　　　　|
    *   | parataxis  | 並列　　　　　　　　　　　　　　|
    *   | punct      | 句読点　　　　　　　　　　　　　|
    *   | reparandu  | 単語として認識されない単語表現　|
    *   | root       | 文の根　　　　　　　　　　　　　|
    *   | vocation   | 発声関係　　　　　　　　　　　　|
    *   | xcomp      | 補体　　　　　　　　　　　　　　|
    *   ----------------------------------------
    * token.head.i: 係受けの相手トークン番号
    * token.head.text: 係受けの相手テキスト
    '''
    annotation = self.nlp(text)
    for sentence in annotation.sents:
      for token in sentence:
        print(
          token.i,
          token.orth_,
          token.lemma_,
          token.norm_,
          token.morph.get('Reading'),
          token.pos_,
          token.morph.get('Inflection'),
          token.tag_,
          token.dep_,
          token.head.i,
        )
      print('EOS')

  def _get_token_syntaxes(self, text: str, symbols: Union[list[str], None]=None) -> list[tuple]:
    doc = self.nlp(text)
    dependencies = []
    for sent in doc.sents:
      for token in sent:
        if symbols is None:
          dependencies.append((token, token.dep_, token.head, token.head.i))
        else:
          if toke.dep_ in symbols:
            dependencies.append((token, token.dep_, token.head, token.head.i))
    return dependencies

  def get_all_token_syntaxes(self, text: str) -> list[tuple]:
    dependencies = self._get_token_syntaxes(text=text, symbols=None)
    return dependencies

  def get_subject_token_syntaxes(self, text: str) -> list[tuple]:
    dependencies = self._get_token_syntaxes(text=text, symbols=['nsubj', 'iobj'])
    return dependencies

  def convert_token_pos_UID_to_jp(self, uid: Union[str, None]=None) -> str:
    uid_to_jp = {
      'ADJ': '形容詞',
      'ADP': '接置詞',
      'ADV': '副詞',
      'AUX': '助動詞',
      'CCONJ': '接続詞',
      'DET': '限定詞',
      'INTJ': '感嘆符',
      'NOUN': '名詞',
      'NUM': '数詞',
      'PART': '助詞',
      'PRON': '固有名詞',
      'PROPN': '代名詞',
      'PUNCT': '句読点',
      'SCONJ': '従属接続詞',
      'SYM': '記号',
      'VERB': '動詞',
      'X': 'その他',
    }
    return uid_to_jp if uid is None else uid_to_jp[uid.upper()]

  def convert_token_dep_UID_to_jp(self, uid: Union[str, None]=None) -> str:
    uid_to_jp = {
      'acl': '名詞節修飾語',
      'advcl': '副詞節修飾語',
      'advmod': '副詞修飾語',
      'amod': '形容詞修飾語',
      'appos': '同格',
      'aux': '助動詞',
      'case': '格表現',
      'cc': '等位接続詞',
      'ccomp': '捕文',
      'clf': '類別詞',
      'compound': '複合名詞',
      'conj': '結合詞',
      'cop': '連結詞',
      'csubj': '主部',
      'dep': '不明な依存関係',
      'det': '限定詞',
      'discourse': '談話要素',
      'dislocated': '転置',
      'expl': '嘘辞',
      'fixed': '固定複数単語表現',
      'flat': '同格複数単語表現',
      'goeswith': '一単語分割表現',
      'iobj': '間接目的語',
      'list': 'リスト表現',
      'mark': '接続詞',
      'nmod': '名詞修飾語',
      'nsubj': '主語名詞',
      'nummod': '数詞修飾語',
      'obj': '目的語',
      'obl': '斜格名詞',
      'orphan': '独立関係',
      'parataxis': '並列',
      'punct': '句読点',
      'reparandu': '単語として認識されない単語表現',
      'root': '文の根',
      'vocation': '発声関係',
      'xcomp': '補体',
    }
    return uid_to_jp if uid is None else uid_to_jp[uid.lower()]

  # 固有表現抽出
  def print_named_entities(self, text: str) -> None:
    '''
    entの主なプロパティ。
    * ent.text: テキスト
    * ent.label_: ラベル
    * ent.start_char: 開始位置
    * ent.end_char: 終了位置
    '''
    doc = self.nlp(text)
    for ent in doc.ents:
      print(
        ent.text,
        ent.label_,
        ent.start_char,
        ent.end_char,
      )
    print('EOS')

  def add_named_entries(self, rules: list[dict[str, str]]) -> None:
    ruler = self.nlp.add_pipe('entity_ruler')
    ruler.add_patterns(rules)

  def get_named_entries(self, text: str) -> list:
    doc = self.nlp(text)
    return doc.ents

  # 名詞句抽出
  def print_nuon_chunks(self, text: str) -> None:
    doc = self.nlp(text)
    for chunk in doc.noun_chunks:
      print(
        chunk.text
      )
    print('EOS')

  def get_noun_chunks(self, text: str):
    doc = self.nlp(text)
    return doc.noun_chunks

  # データフレーム、可視化
  def get_as_dataframe(self, text: str):
    doc = self.nlp(text)
    # 依存構文解析結果の表形式表示
    results = []
    for sent in doc.sents:
      # 1文ごとに改行表示(センテンス区切り表示)
      # 各文を解析して結果をlistに入れる(文章が複数ある場合も一まとめにする)
      for token in sent:
        info_dict = {}
        info_dict['.i'] = token.i # トークン番号
        info_dict['.orth_'] = token.orth_ # オリジナルテキスト
        info_dict['._.reading'] = token._.reading # 読み仮名
        info_dict['.pos_'] = token.pos_ # 品詞(UID)
        info_dict['.tag_'] = token.tag_ # 品詞(日本語)
        info_dict['.lemma_'] = token.lemma_ # 基本形(名寄せ後)
        info_dict['._.inf'] = token._.info # 活用情報
        info_dict['.rank'] = token.rank # 頻度のように扱える?
        info_dict['.norm_'] = token.norm_ # 原型
        info_dict['.is_oov'] = token.is_oov # 登録されていない単語か?
        info_dict['.is_stop'] = token.is_stop # ストップワードか?
        info_dict['.has_vector'] = token.has_vector # word2vecの情報を持っているか?
        info_dict['list(.lefts)'] = list(token.lefts) # 関連語(左)
        info_dict['list(.rights)'] = list(token.rights) # 関連語(右)
        info_dict['.dep_'] = token.dep_ # 係受けの関連性
        info_dict['.head.i'] = token.head.i # 係受けの相手トークン番号
        info_dict['.head.text'] = token.head.text # 係受けの相手テキスト
        results.append(info_dict)

    if 'pandas' in sys.modules:
      results = pd.DataFrame(results)
    return results

  def display_dependencies(self, text: str, port: int=5001):
    doc = self.nlp(text)
    displacy.serve(doc, style='dep', port=port)

  def display_entries(self, text: str, port: int=5002):
    doc = self.nlp(text)
    displacy.serve(doc, style='ent', port=port)

  def display_token_parts_of_speech(self, text: str, plot_name: str):
    doc = self.nlp(text)

    pos = []
    for sent in doc.sents:
      for token in sent:
        pos.append(token.pos_)
    pos_counts = {self.convert_token_pos_UID_to_jp(uid=x): pos.count(x) for x in set(pos)}

    plt.figure()
    plt.bar(dep_counts.keys(), dep_counts.values(), color='darkorange')
    plt.title('テキスト内で見つかった依存関係')
    plt.xticks(rotation=90)
    plt.xlabel('依存関係')
    plt.ylabel('見つかった数')
    plt.grid(True)
    plt.subplots_adjust(bottom=0.33)
    plt.savefig(plot_name)

  def display_token_dependencies(self, text: str, plot_name: str):
    doc = self.nlp(text)

    dep = []
    for sent in doc.sents:
      for token in sent:
        dep.append(token.dep_)
    dep_counts = {self.convert_token_dep_UID_to_jp(uid=x): dep.count(x) for x in set(dep)}

    plt.figure()
    plt.bar(dep_counts.keys(), dep_counts.values(), color='darkorange')
    plt.title('テキスト内で見つかった依存関係')
    plt.xticks(rotation=90)
    plt.xlabel('依存関係')
    plt.ylabel('見つかった数')
    plt.grid(True)
    plt.subplots_adjust(bottom=0.33)
    plt.savefig(plot_name)

  def display_token_pos_connections(self, text: str, plot_name: str):
    doc = self.nlp(text)

    pos_from = []
    pos_to = []
    for sent in doc.sents:
      for token in sent:
        pos_from.append(self.convert_token_pos_UID_to_jp(token.pos_))
        pos_to.append(self.convert_token_pos_UID_to_jp(tokne.head.pos_))

    mapping_pos_from = {val: i for i, val in enumerate(sorted(set(pos_from)))}
    mapping_pos_to = {val: i for i, val in enumerate(sorted(set(pos_to)))}

    numeric_pos_from = [mapping_pos_from[val] for val in pos_from]
    numeric_pos_to = [mapping_pos_to[val] for val in pos_to]

    bin_label_pos_from = sorted(set(pos_from))
    bin_label_pos_to = sorted(set(pos_to))

    plt.figure()
    plt.hist2d(numeric_pos_to, numeric_pos_from, bins=(len(mapping_pos_to), len(mapping_pos_from)), cmap='plasma')
    plt.colorbar(label='頻度')
    plt.title('係受けの構造')
    plt.xticks(range(len(bin_label_pos_to)), bin_label_pos_to, rotation=90)
    plt.yticks(range(len(bin_label_pos_from)), bin_label_pos_from, rotation=0)
    plt.xlabel('係受け元の品詞')
    plt.ylabel('係受け先の品詞')
    plt.grid()
    plt.subplots_adjust(left=0.20, bottom=0.33)
    plt.savefig(plot_name)

if __name__ == '__main__':
  parser = GiNZANaturalLanguageProcessing()
  # sentences = parser.get_sentences(text='この商品はよく効きます。この商品はよく売れます。')
  # bunsetu = parser.get_bunsetu_spans(text='この商品はよく効きます。この商品はよく売れます。')
  # bunsetu_phrase = parser.get_bunsetu_phrase_spans(text='この商品はよく効きます。この商品はよく売れます。')
  # bunsetu_dependencies = parser.get_bunsetu_syntaxes(text='この商品はよく効きます。この商品はよく売れます。')

  # parser.print_token_syntaxes(text='昨日から胃がキリキリと痛い。ただ、熱は無い。')
  # parser.print_token_syntaxes(text='No.1にならなくても良い、もともと特別なオンリーワン。')
  # subject_list = parser.get_all_token_syntaxes(text='この商品はよく効きます。この商品はよく売れます。')

  # parser.add_named_entries(
  #   rules=[
  #     {'label': 'Person', 'pattern': 'サツキ'},
  #     {'label': 'Person', 'pattern': 'メイ'},
  #   ]
  # )
  # parser.print_named_entities(
  #   text='小学生のサツキと妹のメイは、母の療養のために父と一緒に初夏の頃の農村へ引っ越してくる。'
  # )
  # entries = parser.get_named_entries(text='小学生のサツキと妹のメイは、母の療養のために父と一緒に初夏の頃の農村へ引っ越してくる。')
  # parser.print_noun_chunks(text='錦織圭選手は偉大なテニス選手です。')