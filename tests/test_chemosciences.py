import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import chemosciences_digest as c


CONFIG = c.read_json(c.ROOT / 'config' / 'chemosciences.json')


def paper(title, abstract='', pmid='1', **kw):
    return dict(pmid=pmid, title=title, abstract=abstract, authors=['Unknown A'], journal='Small journal',
                pubdate='2026-10-01', doi='', types=[], preprint=False, url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', **kw)


class OntologyTests(unittest.TestCase):
    def test_no_gene_substrings(self):
        self.assertFalse(c.matches('secreted receptor', 'ret'))
        self.assertTrue(c.matches('RET signaling', 'ret'))
        self.assertTrue(c.matches('taste-bud cells', 'taste bud*'))

    def test_repeat_words_do_not_inflate_score(self):
        a=paper('Olfactory progenitor differentiation', 'Tuft cells and IL-13 regulate epithelial regeneration.')
        b=copy.deepcopy(a);b['abstract'] *= 100
        self.assertEqual(c.score_paper(a,CONFIG)['score'],c.score_paper(b,CONFIG)['score'])

    def test_prestige_and_affiliation_do_not_change_score(self):
        a=paper('Taste receptor ligand binding', 'In vitro receptor pharmacology measures calcium signaling.')
        b={**a,'authors':['Famous PI'],'journal':'Famous journal','affiliation':'Monell'}
        self.assertEqual(c.score_paper(a,CONFIG),c.score_paper(b,CONFIG))

    def test_computational_and_clinical_work_qualify(self):
        for p in [paper('Dynamics of ligand binding in a bitter taste receptor','Molecular dynamics simulation tests receptor gating and taste perception.'),
                  paper('Predicting taste alteration after radiotherapy','A clinical cohort measures taste loss in patients.')]:
            self.assertTrue(c.score_paper(p,CONFIG)['eligible'])

    def test_incidental_mention_and_gas_sensor_do_not_qualify(self):
        for p in [paper('Frontal cortex belief updating','Olfactory areas were controls for a neuronal study.'),
                  paper('Artificial olfactory chip gas sensor','An electronic sensor detects benzene without biological validation.')]:
            self.assertFalse(c.score_paper(p,CONFIG)['eligible'])

    def test_corrections_and_retractions_not_promoted(self):
        p=paper('Olfactory progenitor regeneration','Stem cell lineage signaling in olfactory epithelium.')
        p['types']=['Retracted Publication']
        self.assertFalse(c.score_paper(p,CONFIG)['eligible'])

    def test_selection_caps_reviews_and_does_not_fill_with_noise(self):
        p=paper('Olfactory progenitor regeneration','Stem cell lineage and immune signaling in the epithelium.')
        data=[{**p,'pmid':str(i),'types':['Review']} for i in range(8)]
        data.append(paper('Benzene gas sensor','Materials only.',pmid='99'))
        r=c.rank_records(data,CONFIG)
        self.assertEqual(sum(x['editor_pick'] for x in r),1)
        self.assertFalse(next(x for x in r if x['paper']['pmid']=='99')['reading_list'])

    def test_fetch_failure_does_not_write_partial_run(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as temp, mock.patch.object(c,'request',side_effect=TimeoutError):
            with self.assertRaises(TimeoutError): c.retrieve(CONFIG,'2026-10-19','2026-10-25')
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_no_silent_search_truncation(self):
        response=json.dumps({'esearchresult':{'count':'10000','idlist':[]}}).encode()
        with mock.patch.object(c,'request',return_value=response):
            with self.assertRaises(RuntimeError):c.retrieve(CONFIG,'2026-10-19','2026-10-25')

    def test_xml_preprint_authors_and_version_relations(self):
        raw=b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Olfactory <i>stem</i> cells</ArticleTitle><Journal><Title>bioRxiv</Title><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal><Abstract><AbstractText>Progenitor regeneration.</AbstractText></Abstract><AuthorList><Author><LastName>Smith</LastName><Initials>AB</Initials></Author></AuthorList><PublicationTypeList><PublicationType>Preprint</PublicationType></PublicationTypeList></Article><CommentsCorrectionsList><CommentsCorrections RefType="UpdateIn"><PMID>456</PMID></CommentsCorrections></CommentsCorrectionsList></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.test/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'''
        p=c.parse_articles(raw)[0]
        self.assertTrue(p['preprint']);self.assertEqual(p['authors'],['Smith AB'])
        self.assertEqual(p['title'],'Olfactory stem cells');self.assertEqual(p['related'][0]['pmid'],'456')

    def test_history_and_network_accumulate_without_touching_phd(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as temp:
            out=Path(temp)/'chemosciences';phd=Path(temp)/'2026-10-26'/'digest.json'
            phd.parent.mkdir();phd.write_text('phd sentinel')
            a=paper('Olfactory progenitor regeneration','Stem cells differentiate in an epithelial niche.')
            c.generate([a],CONFIG,'2026-10-12','2026-10-18','2026-10-19',out)
            first=(out/'2026-10-19'/'digest.json').read_bytes()
            c.generate([{**a,'pmid':'2'}],CONFIG,'2026-10-19','2026-10-25','2026-10-26',out)
            self.assertEqual(first,(out/'2026-10-19'/'digest.json').read_bytes())
            self.assertEqual(len(c.read_json(out/'network.json')['nodes']),2)
            self.assertEqual(phd.read_text(),'phd sentinel')


if __name__=='__main__':unittest.main()
