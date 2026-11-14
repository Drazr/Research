import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.common.data import canonical_id, read_targets, read_original_captions, splits_and_support, REPO
from src.common.contracts import ContractError

class DataTests(unittest.TestCase):
    def test_lossless_ids(self):
        self.assertEqual(canonical_id('AbC'), '0000000000000abc')
        for bad in (123,1.5,'1.5','0x12','1'*17,'../a',' abc'):
            with self.assertRaises(ValueError): canonical_id(bad)

    def test_labels_unknown_and_duplicates(self):
        p = json.loads((REPO/'configs/protocol.json').read_text())
        iid='6a20677eee04fd08'
        row=[iid]+['']*25;row[1+p['category_order'].index('plant.n.02')]='4';row[1]='1'
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'labels.csv'
            def write(rows):
                s=io.StringIO();csv.writer(s).writerows(rows);f.write_bytes(s.getvalue().encode('utf-8'))
            header=['image_id']+p['category_order']
            write([header,row]);t=read_targets(f,p)[iid]
            self.assertEqual(t[0],1);self.assertIsNone(t[p['category_order'].index('plant.n.02')]);self.assertEqual(t[1],0)
            write([header,row,row])
            with self.assertRaises(ValueError):read_targets(f,p)
            row[2]='4';write([header,row])
            with self.assertRaises(ValueError):read_targets(f,p)

    def test_original_fields_and_text(self):
        text='বাংলা e\u0301'
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'caps.zip'
            row={'image/key':'1','bn':{'caption':[text],'caption/tokenized':['wrong']}}
            with zipfile.ZipFile(f,'w') as z:z.writestr('captions.jsonl',json.dumps(row))
            self.assertEqual(read_original_captions(f,['bn'])['0000000000000001']['bn'],[text])
            with zipfile.ZipFile(f,'w') as z:z.writestr('captions.jsonl',json.dumps(row)+'\n'+json.dumps(row))
            with self.assertRaises(ValueError):read_original_captions(f,['bn'])

    def test_incomplete_population_needs_protocol(self):
        p=json.loads((REPO/'configs/protocol.json').read_text())
        with self.assertRaises(ContractError):splits_and_support({'0000000000000001':[0]*25},p)

if __name__=='__main__':unittest.main()
