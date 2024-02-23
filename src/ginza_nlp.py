
# Standard modules
import os
import sys
import pprint
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
    self.doc = None

  def set_doc(self, text: str) -> None:
    self.doc = self.nlp(text)

  ### 文境界解析 ###
  def get_sentences(self, text: Union[str, None]=None) -> list[str]:
    if not text is None: self.doc = self.nlp(text)
    return self.doc.sents

  ### 文節 ###
  def get_bunsetu_spans(self, text: Union[str, None]=None) -> list[str]:
    if not text is None: self.doc = self.nlp(text)
    return ginza.bunsetu_spans(self.doc)

  def get_bunsetu_phrase_spans(self, text: Union[str, None]) -> list[str]:
    if not text is None: self.doc = self.nlp(text)
    return ginza.bunsetu_phrase_spans(self.doc)

  ### 形態素解析 ###
  def print_token_syntaxes(self, text: Union[str, None]=None) -> None:
    '''
    https://qiita.com/kei_0324/items/400f639b2f185b39a0cf
    https://spacy.io/api/token
    https://www.anlp.jp/proceedings/annual_meeting/2015/pdf_dir/E3-4.pdf
    * token.i: トークン番号
    * token.orth_: オリジナルテキスト
    * token._.reading: 読み仮名
    * token.pos_: 品詞(UID)
    *   ----------------------------------------------------------------------------------------------------------------------------------------------
    *   | UID   | 日本語名　　　　　　　　　　　　　　　　　　　　　　　　　| 説明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 例　　　　　　　　　　　　　|
    *   ----------------------------------------------------------------------------------------------------------------------------------------------
    *   | NOUN  | 名詞ー普通名詞　　　　　　　　　　　　　　　　　　　　　　| 物体、物質、人名、場所など　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 水、犬、東京　　　　　　　　|
    *   | PROPN | 名詞ー固有名詞　　　　　　　　　　　　　　　　　　　　　　| 個人名や場所の名前など　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 私、これ、そこ　　　　　　　|
    *   | VERB  | 動詞、名詞ーサ変可能で動詞の語尾がついたもの　　　　　　　| 物事の動作や作用、状態、存在などを示す　　　　　　　　　　　　　　　　　　　　　　　| 動く、食べる、咲く　　　　　|
    *   | ADJ   | 形容詞、連体詞、名詞ー形容詞可能で形容詞の語尾がつく場合　| 名詞や代名詞を修飾する　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 暑い、親切な　　　　　　　　|
    *   | ADV   | 副詞　　　　　　　　　　　　　　　　　　　　　　　　　　　| 動詞、形容詞、ほかの副詞や分全体を修飾する　　　　　　　　　　　　　　　　　　　　　| すっかり、ずっと　　　　　　|
    *   | INTJ  | 感動詞　　　　　　　　　　　　　　　　　　　　　　　　　　| 「！」　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| あっ　　　　　　　　　　　　|
    *   ----------------------------------------------------------------------------------------------------------------------------------------------
    *   | PUNCT | 補助記号ー句点、読点、括弧開、括弧閉　　　　　　　　　　　| 「。」「、」　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | SYM   | 記号、補助記号のうちPUNCT以外　　 　　　　　　　　　　　| 「？」　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | X     | 空白　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   ----------------------------------------------------------------------------------------------------------------------------------------------
    *   | PRON  | 代名詞　　　　　　　　　　　　　　　　　　　　　　　　　　| 名詞または名詞句の代わりに用いられる　　　　　　　　　　　　　　　　　　　　　　　　| 私、これ、それ　　　　　　　|
    *   | NUM   | 名詞ー数詞　　　　　　　　　　　　　　　　　　　　　　　　| ０、１０００　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　|
    *   | AUX   | 助動詞、動詞・形容詞のうち非自立なもの　　　　　　　　　　| 主語や動詞などと一緒に使われ、動詞だけでは表現できない文の意味や時制などを表現する　| れる、らしい　　　　　　　　|
    *   | CONJ  | 接続詞、助詞ー接続助詞のうち等位接続詞　　　　　　　　　　| 分の構成要素同士の関係を示す　　　　　　　　　　　　　　　　　　　　　　　　　　　　| また、そして、しかし　　　　|
    *   | SCONJ | 接続詞、助詞ー接続助詞、準体助詞　　　　　　　　　　　　　| 主節の補足説明をする　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| なぜなら、もし　　　　　　　|
    *   | DET   | 連体詞の一部　　　　　　　　　　　　　　　　　　　　　　　| 名詞をより明確に示す　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 一つの　　　　　　　　　　　|
    *   | ADP   | 助詞ー格助詞、副助詞、係助詞　　　　　　　　　　　　　　　| 名詞句と結びつき、文中のほかの要素との関連を示す　　　　　　　　　　　　　　　　　　| 〜が、〜へ　　　　　　　　　|
    *   | PART  | 助詞ー終助詞、接尾詞　　　　　　　　　　　　　　　　　　　| 言葉に意味を肉付けする　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 〜を、〜が　　　　　　　　　|
    *   ----------------------------------------------------------------------------------------------------------------------------------------------
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
    *   ------------------------------------------------------------------------------------------------------
    *   | 大分類　 　　| tag            | 説明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   ------------------------------------------------------------------------------------------------------
    *   | 述語の要素　 | nsubj          | 主格で述語に係る名詞句。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | nsubjpass      | 主格で受身の助動詞を伴う用言に係る名詞句。　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | dobj           | 目的格で述語に係る名詞句。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | iobj           | 格助詞「に」を伴うなどして述語に係る名詞句。　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | nmod           | これまでに示した以外の格の名詞句や、時相名詞により用言を修飾する場合。　　　　　　　　　　　　|
    *   | 　　　　　　 | csubj          | 主語になる名詞節。準体助詞を伴う用言句が主語となる場合。　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | ccomp          | 補文。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | advcl          | 副詞節。主に接続助詞をともなって用言を修飾する節。　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | advmod         | 副詞による修飾。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | neg            | 否定語の付与。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   ------------------------------------------------------------------------------------------------------
    *   | 名詞の修飾　 | nummod         | 数量の指定。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | appos          | 同格の表現。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | acl            | 連体修飾節。ただしａｍｏｄに該当する場合を除く。この他「てからの」「ながらの」などの接続表現。|
    *   | 　　　　　　 | amod           | 形容詞、形状詞、連体詞（DET以外）が格を伴わずに名詞を修飾する場合。　  　　　　　　　　　　 |
    *   | 　　　　　　 | det            | DETによる修飾。　　　　　　　　　　　　　　　　　　　　　　　　　　　　  　　　　　　　　　 |
    *   ------------------------------------------------------------------------------------------------------
    *   | 複合語　　　 | compound       | 名詞と名詞・動詞と動詞の複合。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | name           | 固有名詞の複合語。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | mwe            | 機能表現の複合語。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | foreign        | 外国語の複合語。常に左側を主辞とする。　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   ------------------------------------------------------------------------------------------------------
    *   | 並列　　　　 | conj           | 並列構造。左側の要素を主辞とする。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | cc             | 等位接続詞。「アダムとイブ」の「と」の部分。　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   ------------------------------------------------------------------------------------------------------
    *   | その他の要素 | aux            | 用言に付く助動詞や、非自立の補助用言。「か」などの終助詞を含む。　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | cop            | 繋辞の「だ」「です」が付く場合。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | mark           | 従属接続詞、接続助詞、構文標識の「と」「か」などが付く場合。　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | case           | 助詞による格の表現。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   | 　　　　　　 | punct          | 句読点。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
    *   ------------------------------------------------------------------------------------------------------
    *   | xcomp, evalなどのの日本語に無いもの、goeswith, vocative, list, remnantなどの特殊ラベルは割愛。　　　　　　　　　　　　|
    *   ------------------------------------------------------------------------------------------------------
    * token.head.i: 係受けの相手トークン番号
    * token.head.text: 係受けの相手テキスト
    '''
    if not text is None: self.doc = self.nlp(text)
    for sent in self.doc.sents:
      for token in sent:
        print(
          token.i,
          token.text,
          # token.orth_,
          token.lemma_,
          # token.norm_,
          # token.morph.get('Reading'),
          token.pos_,
          # token.morph.get('Inflection'),
          token.tag_,
          token.dep_,
          # self.convert_token_dep_UID_to_jp(token.dep_),
          token.is_stop,
          token.head.i,
          token.head.text,
          token.head.pos_,
        )
      print('EOS')

  def _get_tokens(self, text: Union[str, None]=None, symbols: Union[list[str], None]=None) -> list:
    if not text is None: self.doc = self.nlp(text)
    tokens = []
    for sent in doc.sents:
      for token in sent:
        if symbols is None:
          tokens.append(token)
        else:
          if token.pos_ in symbols:
            tokens.append(token)
    return tokens

  def get_tokens(self, text: Union[str, None]=None) -> list:
    return self._get_tokens(text=text, symbols=None)

  def get_noun_tokens(self, text: Union[str, None]=None) -> list:
    # 名詞を取得
    return self._get_tokens(text=text, symbols=['NOUN', 'PROPN', 'PRON'])

  def get_verb_tokens(self, text: Union[str, None]=None) -> list:
    # 動詞を取得
    return self._get_tokens(text=text, symbols=['VERB'])

  def get_adjective_tokens(self, text: Union[str, None]=None) -> list:
    # 形容詞を取得
    return self._get_tokens(text=text, symbols=['ADJ'])

  def get_adverb_tokens(self, text: Union[str, None]=None) -> list:
    # 副詞を取得
    return self._get_tokens(text=text, symbols=['ADV'])

  def get_numeral_tokens(self, text: Union[str, None]=None) -> list:
    # 数詞を取得
    return self._get_tokens(text=text, symbols=['NUM'])

  def get_auxiliary_verb_tokens(self, text: Union[str, None]=None) -> list:
    # 助動詞を取得
    return self._get_tokens(text=text, symbols=['AUX'])

  def get_conjunction_tokens(self, text: Union[str, None]=None) -> list:
    # 接続詞を取得
    return self._get_tokens(text=text, symbols=['CONJ', 'SCONJ'])

  def get_postpositional_particle_tokens(self, text: Union[str, None]=None) -> list:
    # 助詞を取得
    return self._get_tokens(text=text, symbols=['ADP', 'PART'])

  def get_meaningful_tokens(self, text: Union[str, None]=None) -> list:
    # 意味を持つ(ROOTになり得る)品詞を取得
    tokens = self._get_tokens(text=text, symbols=['NOUN', 'PROPN', 'PRON', 'VERB', 'ADJ', 'ADV'])
    tokens = [token for token in tokens if not token.head.pos_ in ['NOUN', 'PROPN', 'PRON']]
    return tokens

  def _get_tokens_except(self, text: Union[str, None]=None, symbols: Union[list[str], None]=None) -> list:
    if not text is None: self.doc = self.nlp(text)
    tokens = []
    for sent in self.doc.sents:
      for token in sent:
        if symbols is None:
          tokens.append(token)
        else:
          if not token.pos_ in symbols:
            tokens.append(token)
    return tokens

  def get_tokens_except_noun(self, text: Union[str, None]=None) -> list:
    # 名詞以外を取得
    return self._get_tokens_except(tex=text, symbols=['NOUN', 'PROPN', 'PRON'])

  def get_tokens_except_verb(self, text: Union[str, None]=None) -> list:
    # 動詞以外を取得
    return self._get_tokens_except(text=text, symbols=['VERB'])

  def get_tokens_except_adjective(self, text: Union[str, None]=None) -> list:
    # 形容詞以外を取得
    return self._get_tokens_except(text=text, symbols=['ADJ'])

  def get_tokens_except_adverb(self, text: Union[str, None]=None) -> list:
    # 副詞以外を取得
    return self._get_tokens_except(text=text, symbols=['ADV'])

  def get_tokens_except_numeral(self, text: Union[str, None]=None) -> list:
    # 数詞以外を取得
    return self._get_tokens_except(text=text, symbols=['NUM'])

  def get_tokens_except_auxiliary_verb(self, text: Union[str, None]=None) -> list:
    # 助動詞以外を取得
    return self._get_tokens_except(text=text, symbols=['AUX'])

  def get_tokens_except_conjunction(self, text: Union[str, None]=None) -> list:
    # 接続詞以外を取得
    return self._get_tokens_except(text=text, symbols=['CONJ', 'SCONJ'])

  def get_tokens_except_postpositional_particle(self, text: Union[str, None]=None) -> list:
    # 助詞以外を取得
    return self._get_tokens_except(text=text, symbols=['ADP', 'PART'])

  def get_tokens_except_meaningless(self, text: Union[str, None]=None) -> list:
    return self._get_tokens_except(text=text, symbols=['INTJ', 'PUNCT', 'SYM', 'X', 'NUM', 'AUX', 'CONJ', 'SCONJ', 'DET', 'ADP', 'PART'])

  def _get_token_syntaxes(self, text: Union[str, None]=None, symbols: Union[list[str], None]=None) -> list[tuple]:
    if not text is None: self.doc = self.nlp(text)
    dependencies = []
    for sent in self.doc.sents:
      for token in sent:
        if symbols is None:
          dependencies.append((token, token.head))
        else:
          if token.dep_ in symbols:
            dependencies.append((token, token.head))
    return dependencies

  def get_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    return self._get_token_syntaxes(text=text, symbols=None)

  def get_root_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    # 文の根に該当するトークン取得
    return self._get_token_syntaxes(text=text, symbols=['root', 'ROOT'])

  def get_predicate_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    # 述語を修飾するトークンの取得
    return self._get_token_syntaxes(text=text, symbols=['nsubj', 'nsubjpass', 'dobj', 'iobj', 'nmod', 'csubj', 'ccomp', 'advcl', 'advmod', 'neg'])

  def get_noun_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    # 名詞を修飾するトークンの取得
    return self._get_token_syntaxes(text=text, symbols=['nummod', 'appos', 'acl', 'amod', 'det'])

  def get_compound_word_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    # 複合語を形成するトークンの取得
    return self._get_token_syntaxes(text=text, symbols=['compound', 'name', 'mwe', 'foreign'])

  def get_parallel_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    # 並列関係を構築するトークンの取得
    return self._get_token_syntaxes(text=text, symbols=['conj', 'cc'])

  def get_other_token_syntaxes(self, text: Union[str, None]=None) -> list[tuple]:
    # その他の要素を修飾するトークンの取得
    return self._get_token_syntaxes(text=text, symbols=['aux', 'cop', 'mark', 'case', 'punct'])

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

  ### 固有表現抽出 ###
  def print_named_entities(self, text: Union[str, None]=None) -> None:
    '''
    entの主なプロパティ。
    * ent.text: テキスト
    * ent.label_: ラベル
    * ent.start_char: 開始位置
    * ent.end_char: 終了位置
    '''
    if not text is None: self.doc = self.nlp(text)
    for ent in doc.ents:
      print(
        ent.text,
        ent.orth_,
        ent.lemma_,
        ent.label_,
        ent.start_char,
        ent.end_char,
      )
    print('EOS')

  def add_named_entities(self, rules: list[dict[str, str]]) -> None:
    ruler = self.nlp.add_pipe('entity_ruler')
    ruler.add_patterns(rules)

  def get_named_entities(self, text: Union[str, None]=None) -> list:
    if not text is None: self.doc = self.nlp(text)
    return self.doc.ents

  ### 名詞句抽出 ###
  def print_noun_chunks(self, text: Union[str, None]=None) -> None:
    if not text is None: self.doc = self.nlp(text)
    for chunk in self.doc.noun_chunks:
      print(
        chunk.text,
        chunk.orth_,
        chunk.lemma_,
        chunk.root.text,
        chunk.root.head.i,
        chunk.root.head.text,
        chunk.root.dep_,
      )
    print('EOS')

  def get_noun_chunks(self, text: Union[str, None]=None):
    if not text is None: self.doc = self.nlp(text)
    return self.doc.noun_chunks

  ### 係受け解析 ###
  def get_nth_depth_token_syntaxes(self, text: Union[str, None]=None, nth: int=3) -> list:
    tokens = self.get_tokens(text=text)
    nth_depth_token_syntaxes = {}
    for token in tokens:
      _nth_depth_token_syntaxes = []
      _token = token.head
      ith = 0
      while ith < nth:
        _nth_depth_token_syntaxes.append(_token)
        if _token == _token.head: break
        _token = _token.head
        ith += 1
      nth_depth_token_syntaxes[token] = _nth_depth_token_syntaxes
    return nth_depth_token_syntaxes

  def get_full_depth_token_syntaxes(self, text: Union[str, None]=None) -> list:
    return self.get_nth_depth_token_syntaxes(text=text, nth=len(text))

  def get_nth_depth_named_entity_syntaxes(self, text: Union[str, None]=None, nth: int=3) -> list:
    entities = self.get_named_entities(text=text)
    nth_depth_entity_syntaxes = {}
    for entity in entities:
      _nth_depth_entity_syntaxes = []
      _entity = entry.root.head
      ith = 0
      while ith < nth:
        _nth_depth_entity_syntaxes.append(_entry)
        if _entity == _entity.head: break
        _entry = _entry.head
        ith += 1
      nth_depth_entity_syntaxes[entity] = _nth_depth_entity_syntaxes
    return nth_depth_entity_syntaxes

  def get_full_depth_named_entity_syntaxes(self, text: Union[str, None]=None) -> list:
    return self.get_nth_depth_named_entity_syntaxes(text=text, nth=len(text))

  def get_nth_depth_noun_chunk_syntaxes(self, text: Union[str, None]=None, nth: int=3) -> list:
    chunks = self.get_noun_chunks(text=text)
    nth_depth_chunk_syntaxes = {}
    for chunk in chunks:
      _nth_depth_chunk_syntaxes = []
      _chunk = chunk.root.head
      ith = 0
      while ith < nth:
        _nth_depth_chunk_syntaxes.append(_chunk)
        if _chunk == _chunk.head: break
        _chunk = _chunk.head
        ith += 1
      nth_depth_chunk_syntaxes[chunk] = _nth_depth_chunk_syntaxes
    return nth_depth_chunk_syntaxes

  def get_full_depth_noun_chunk_syntaxes(self, text: Union[str, None]=None) -> list:
    return self.get_nth_depth_noun_chunk_syntaxes(text=text, nth=len(text))

  ### 否定表現判定 ###
  def check_negative_sentence(self, text: Union[str, None]=None) -> bool:
    if not text is None: self.doc = self.nlp(text)
    for sent in self.doc.sents:
      for token in sent:
        ## 条件を列挙し、それに当てはまる場合にTrueそうでなければFalseを返す。 ##
        # ex) メイクもスッキリ落ちて洗い上がりもぬるぬる残ら「ない」。
        if token.lemma_ == 'ない' and token.pos_ == 'AUX' and token.is_stop and token.head.pos_ == 'VERB':
          return True
        # ex) 使用後はつっぱることも「なく」肌がふっくらする
        if token.lemma_ == 'ない' and token.pos_ == 'AUX' and token.is_stop and token.head.pos_ == 'NOUN':
          return True
        # ex) ベタベタし「ない」です
        if token.lemma_ == 'ない' and token.pos_ == 'AUX' and token.is_stop and token.head.pos_ == 'ADV':
          return True
        # ex) お肌が柔らかくモチモチのくすみの「ない」肌に。
        if token.lemma_ == 'ない' and token.pos_ == 'ADJ' and token.is_stop and token.head.pos_ == 'NOUN':
          return True
        # ex) 肌のゴワつきも「なく」なるの
        if token.lemma_ == 'ない' and token.pos_ == 'ADJ' and token.is_stop and token.head.pos_ == 'VERB':
          return True
        # ex) つっぱる感じは「なかっ」たです。
        if token.lemma_ == 'ない' and token.pos_ == 'ADJ' and token.is_stop and token.head.pos_ == 'ADJ':
          return True
        # ex) 全くお肌にトラブルが「なく」、スッキリ、しっとりな洗い上がり
        if token.lemma_ == 'ない' and token.pos_ == 'ADJ' and token.is_stop and token.head.pos_ == 'ADV':
          return True
        # ex) 肌荒れしませ「ん」。
        if token.lemma_ == 'ぬ' and token.pos_ == 'AUX' and token.is_stop and token.head.pos_ == 'VERB':
          return True
        # ex) 洗った後に全く突っ張ら「ず」。
        if token.lemma_ == 'ず' and token.pos_ == 'AUX' and token.is_stop and token.head.pos_ == 'VERB':
          return True
        # ex) オイルなのにヌルヌルせ「ず」
        if token.lemma_ == 'ず' and token.pos_ == 'AUX' and token.is_stop and token.head.pos_ == 'ADV':
          return True
        # ex) こちらを使用して、肌荒れが起こりにくくなりました
        if token.lemma_ == 'にくい' and token.pos_ == 'AUX' and token.head.pos_ == 'VERB':
          return True
        # ex) 毛穴のザラザラ感が「なくなり」ました。
        if token.lemma_ in ['なくなる', '無くなる'] and token.pos_ == 'VERB':
          return True

    return False

  ### データフレーム化 ###
  def get_as_dataframe(self, text: Union[str, None]=None):
    if not text is None: self.doc = self.nlp(text)
    # 依存構文解析結果の表形式表示
    results = []
    for sent in self.doc.sents:
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

  ### 可視化 ###
  def display_dependencies(self, text: Union[str, None]=None, port: int=5001):
    if not text is None: self.doc = self.nlp(text)
    displacy.serve(self.doc, style='dep', port=port, options={
      'compact': True,
      'color': '#000000',
      'bg': '#ffffff',
    })

  def display_entities(self, text: Union[str, None]=None, port: int=5002):
    if not text is None: self.doc = self.nlp(text)
    displacy.serve(self.doc, style='ent', port=port, options={
      'compact': True,
      'color': '#000000',
      'bg': '#ffffff',
    })

  def display_token_parts_of_speech(self, text: Union[str, None]=None, plot_name: str):
    if not text is None: self.doc = self.nlp(text)

    pos = []
    for sent in self.doc.sents:
      for token in sent:
        pos.append(token.pos_)
    pos_counts = {self.convert_token_pos_UID_to_jp(uid=x): pos.count(x) for x in set(pos)}

    plt.figure()
    plt.bar(pos_counts.keys(), pos_counts.values(), color='darkorange')
    plt.title('テキスト内で見つかった品詞')
    plt.xticks(rotation=90)
    plt.xlabel('品詞')
    plt.ylabel('見つかった数')
    plt.grid(True)
    plt.subplots_adjust(bottom=0.33)
    plt.savefig(plot_name)

  def display_token_dependencies(self, text: Union[str, None]=None, plot_name: str):
    if not text is None: self.doc = self.nlp(text)

    dep = []
    for sent in self.doc.sents:
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

  def display_token_pos_connections(self, text: Union[str, None]=None, plot_name: str):
    if not text is None: self.doc = self.nlp(text)

    pos_from = []
    pos_to = []
    for sent in self.doc.sents:
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

  def display_root_token_parts_of_speech(self, texts: list[str], plot_name: str):
    pos = []
    for text in texts:
      pos += [root_token.pos_ for (root_token, _) in self.get_root_token_syntaxes(text=text)]
    pos_counts = {self.convert_token_pos_UID_to_jp(uid=x): pos.count(x) for x in set(pos)}

    plt.figure()
    plt.bar(pos_counts.keys(), pos_counts.values(), color='hotpink')
    plt.title('ROOTの品詞')
    plt.xticks(rotation=90)
    plt.xlabel('品詞')
    plt.ylabel('見つかった数')
    plt.grid(True)
    plt.subplots_adjust(bottom=0.33)
    plt.savefig(plot_name)

if __name__ == '__main__':
  parser = GiNZANaturalLanguageProcessing()

  text='ハウス食品グループの研究では、1日にクルクミン30mgとビサクロン400μgを12週間摂取し続けた、健康な若い男女の肝機能酵素値がとても低下したと報告されています。'
  # text='この薬を飲むことで、著しく肌年齢が下がることが実験で確認されました。'
  # text='究極の美肌を手に入れるために、弊社の化粧水を毎晩たっぷりお使いください。'
  # text='小学生のサツキと妹のメイは、母の療養のために父と一緒に初夏の頃の農村へ引っ越してくる。'

  # print('-' * 50)
  # tokens = parser.get_noun_tokens(text=text)
  # print([token.orth_ for token in tokens])
  # print([token.lemma_ for token in tokens])
  # print([token.dep_ for token in tokens])
  # print([token.head.text for token in tokens])
  # print('-' * 50)
  # tokens = parser.get_verb_tokens(text=text)
  # print([token.orth_ for token in tokens])
  # print([token.lemma_ for token in tokens])
  # print([token.dep_ for token in tokens])
  # print([token.head.text for token in tokens])
  # print('-' * 50)
  # tokens = parser.get_adjective_tokens(text=text)
  # print([token.orth_ for token in tokens])
  # print([token.lemma_ for token in tokens])
  # print([token.dep_ for token in tokens])
  # print([token.head.text for token in tokens])
  # print('-' * 50)
  # tokens = parser.get_adverb_tokens(text=text)
  # print([token.orth_ for token in tokens])
  # print([token.lemma_ for token in tokens])
  # print([token.dep_ for token in tokens])
  # print([token.head.text for token in tokens])
  # print('-' * 50)
  # tokens = parser.get_numeral_tokens(text=text)
  # print([token.orth_ for token in tokens])
  # print([token.lemma_ for token in tokens])
  # print([token.dep_ for token in tokens])
  # print([token.head.text for token in tokens])
  # print('-' * 50)

  # print('=' * 50)
  # print('-' * 50)
  # syntaxes = parser.get_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)
  # syntaxes = parser.get_root_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)
  # syntaxes = parser.get_predicate_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)
  # syntaxes = parser.get_noun_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)
  # syntaxes = parser.get_compound_word_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)
  # syntaxes = parser.get_parallel_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)
  # syntaxes = parser.get_other_token_syntaxes(text=text)
  # print([token.orth_ for token, _ in syntaxes])
  # print([token.lemma_ for token, _ in syntaxes])
  # print([token.dep_ for token, _ in syntaxes])
  # print([token.text for _, token in syntaxes])
  # print('-' * 50)

  # sys.exit(1)

  # parser.add_named_entities(
  #   rules=[
  #     {'label': 'Person', 'pattern': 'サツキ'},
  #     {'label': 'Person', 'pattern': 'メイ'},
  #     {'label': 'FOOD', 'pattern': 'ナポリタン'},
  #     {'label': 'FOOD', 'pattern': [
  #       {'POS': 'NOUN', 'OP': '+'},
  #       {'TEXT': 'スパゲティ'},
  #     ]},
  #     {'label': 'Campany', 'pattern': 'Apple'},
  #   ]
  # )
  # entities = parser.get_named_entities(text=text)
  # print([ent.text for ent in entities])
  # print('-' * 50)
  # noun_chunks = parser.get_noun_chunks(text=text)
  # print([chunk.text for chunk in noun_chunks])
  # print('-' * 50)

  # parser.print_token_syntaxes(text=text)
  # parser.display_dependencies(text=text)
  # pprint.pprint(parser.get_nth_depth_token_syntaxes(text=text, nth=3))
  # pprint.pprint(parser.get_nth_depth_named_entity_syntaxes(text=text, nth=3))
  # pprint.pprint(parser.get_nth_depth_noun_chunk_syntaxes(text=text, nth=3))

  # parser.display_token_parts_of_speech(text=text, plot_name='pos.png')
  # parser.display_token_dependencies(text=text, plot_name='dep.png')
  parser.display_root_token_parts_of_speech(texts=[text], plot_name='root_pos.png')