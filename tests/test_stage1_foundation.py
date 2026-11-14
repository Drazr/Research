"""Software fixtures only: never evidence about XM3600 or a real VLM."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from src.common.contracts import (ContractError, DatasetBundle, FeatureBatch, FeatureSite,
    LANGUAGES, ModelInput, TargetRecord, decoder_depths, fingerprint, resolve_model, select_caption)
from src.common.storage import publish, verify, within

def row(i=1, language='en', caption='a', split='FIT'):
    return ModelInput(f'{i:016x}', f'datasets/fixture/{i}.jpg', 'a'*64,
                      language, caption, 'Unmodified caption', split)

def model_config():
    return {'checkpoint_id':'fixture-only', 'revision':'a'*40, 'processor_revision':'b'*40,
        'tokenizer_revision':'c'*40, 'chat_template_sha256':'d'*64, 'adapter':'fixture-only',
        'precision':'float32', 'image_policy':{'fixture':True}, 'max_input_tokens':128,
        'software_versions':{'fixture':'unit-test-only'}, 'config_fingerprint':None}

class ContractTests(unittest.TestCase):
    def test_original_unicode_and_canonical_hash(self):
        text='हिन्दी বাংলা తెలుగు'
        self.assertEqual(replace(row(), caption_text=text).caption_text, text)
        self.assertEqual(fingerprint({'a':text,'b':1}), fingerprint({'b':1,'a':text}))

    def test_no_gold_field_and_fit_gate(self):
        with self.assertRaises(TypeError):
            ModelInput(**dict(row().__dict__, labels=[1]*25))
        for split in ['DEV-S','DEV-C','SCREEN','CONFIRM']:
            with self.assertRaises(ContractError): replace(row(), split=split).require_fit()

    def test_unknown_is_not_absent(self):
        target=TargetRecord('0000000000000001','FIT',tuple([0,1,None]+[0]*22))
        self.assertEqual(target.known_mask[:3], (True,True,False))
        for bad in [4,False,0.5]:
            with self.assertRaises(ContractError): TargetRecord(target.image_id,'FIT',tuple([bad]*25))

    def test_caption_choice_is_order_independent(self):
        rows=[row(caption='a'),row(caption='b')]
        self.assertEqual(select_caption(rows),select_caption(rows[::-1]))
        with self.assertRaises(ContractError): select_caption([row(),row(language='hi')])

    def test_complete_shared_four_language_population(self):
        protocol={'category_order':[f'fixture-{n}' for n in range(25)]}
        fp=fingerprint(protocol);dfp='a'*64
        bundle=DatasetBundle([row(language=l) for l in LANGUAGES],
            {row().image_id:TargetRecord(row().image_id,'FIT',tuple([0]*25))},
            protocol['category_order'],dfp,fp,
            {'status':'passed','dataset_fingerprint':dfp,'protocol_fingerprint':fp})
        bundle.validate(protocol)
        bundle.records=bundle.records[:3]
        with self.assertRaises(ContractError): bundle.validate(protocol)

    def test_row_swaps_and_nan_are_rejected(self):
        rows=[row(1),row(2)];site=FeatureSite('decoder',1,'last_caption')
        FeatureBatch([r.row_key for r in rows],{site.key:[[1,2],[3,4]]}).validate(rows,[site])
        for keys,vectors in [([r.row_key for r in rows[::-1]],[[1],[2]]),
                             ([r.row_key for r in rows],[[float('nan')],[2]])]:
            with self.assertRaises(ContractError): FeatureBatch(keys,{site.key:vectors}).validate(rows,[site])

    def test_model_fingerprint_detects_config_changes(self):
        cfg=model_config();a=resolve_model({'models':{'qwen':cfg}},'qwen')
        b=resolve_model({'models':{'qwen':dict(cfg,precision='bfloat16')}},'qwen')
        self.assertNotEqual(a['config_fingerprint'], b['config_fingerprint'])
        with self.assertRaises(ContractError): resolve_model({'models':{'qwen':dict(a,precision='bfloat16')}},'qwen')
        with self.assertRaises(ContractError): resolve_model({'models':{'qwen':dict(cfg,revision='main')}},'qwen')

    def test_decoder_depths(self):
        self.assertEqual(decoder_depths(28),(7,14,21,28))
        self.assertEqual(decoder_depths(1),(1,))

class StorageTests(unittest.TestCase):
    def metadata(self,status='passed'):
        return dict(kind='software-fixture',status=status,run_id='fixture',code_commit='0'*40,
            protocol_fingerprint='a'*64,dataset_fingerprint='b'*64,model_fingerprint=None,
            seeds={},software_versions={'fixture':'only'},upstream_artifacts=[])

    def test_completion_corruption_and_duplicate_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); p=publish(root,'shared/fixture/run',self.metadata(),{'payload.json':b'{}'})
            verify(p,expected={'protocol_fingerprint':'a'*64})
            with self.assertRaises(FileExistsError): publish(root,'shared/fixture/run',self.metadata(),{'payload.json':b'{}'})
            with self.assertRaises(ContractError): verify(p,expected={'protocol_fingerprint':'c'*64})
            (p/'payload.json').write_bytes(b'corruption')
            with self.assertRaises(ContractError): verify(p)

    def test_blocked_is_not_passed_and_partial_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=publish(root,'smoke/fixture/run',self.metadata('blocked'),{'report.json':b'{}'})
            with self.assertRaises(ContractError): verify(p)
            self.assertEqual(verify(p,required_status=None)['status'],'blocked')
            (root/'partial').mkdir()
            with self.assertRaises(ContractError): verify(root/'partial')

    def test_paths_cannot_escape_root(self):
        for bad in ['../elsewhere','C:/elsewhere','/elsewhere','a/../../b','a\\b','']:
            with self.assertRaises(ContractError): within(Path.cwd(),bad)

if __name__=='__main__': unittest.main()
