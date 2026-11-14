"""Source-cache publication without hard-link support; no remote Drive claim."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.common.contracts import ContractError
from src.common.data import fetch_sources, REPO
from src.common.storage import publish, verify, file_sha256

class CacheTests(unittest.TestCase):
    payload = b'original source bytes\n' * 1024

    def protocol(self):
        return json.loads((REPO/'configs/protocol.json').read_text())

    def sources(self):
        return {'fixture.bin': ('https://fixture.invalid/source',hashlib.sha256(self.payload).hexdigest())}

    def test_completed_cache_without_hardlinks_or_network_on_second_read(self):
        with tempfile.TemporaryDirectory() as tmp, patch('src.common.data.SOURCES',self.sources()), \
                patch('src.common.data.urlopen',return_value=io.BytesIO(self.payload)) as network, \
                patch('os.link',side_effect=OSError('hardlinks unsupported')) as links:
            root=Path(tmp)
            a=fetch_sources(root,self.protocol(),'0'*40)
            b=fetch_sources(root,self.protocol(),'0'*40)
            self.assertEqual(a,b);self.assertEqual(a['fixture.bin'].read_bytes(),self.payload)
            self.assertEqual(network.call_count,1);links.assert_not_called()
            verify(a['fixture.bin'].parent)

    def test_partial_cache_ignored_completed_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch('src.common.data.SOURCES',self.sources()), \
                patch('src.common.data.urlopen',return_value=io.BytesIO(self.payload)):
            root=Path(tmp);digest=hashlib.sha256(self.payload).hexdigest()
            partial=root/'cache/shared/source-downloads'/digest/'unfinished'
            partial.mkdir(parents=True);(partial/'fixture.bin').write_bytes(b'partial')
            result=fetch_sources(root,self.protocol(),'0'*40)
            self.assertNotEqual(result['fixture.bin'].parent,partial)
            result['fixture.bin'].write_bytes(b'corrupt')
            with self.assertRaises(ContractError):fetch_sources(root,self.protocol(),'0'*40)
            self.assertEqual((partial/'fixture.bin').read_bytes(),b'partial')

    def test_verified_legacy_source_copied_without_overwrite_or_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch('src.common.data.SOURCES',self.sources()), \
                patch('src.common.data.urlopen',side_effect=AssertionError('must reuse legacy')):
            root=Path(tmp);legacy=root/'cache/shared/source-downloads/fixture.bin'
            legacy.parent.mkdir(parents=True);legacy.write_bytes(self.payload)
            result=fetch_sources(root,self.protocol(),'0'*40)
            self.assertNotEqual(result['fixture.bin'],legacy)
            self.assertEqual(legacy.read_bytes(),self.payload)
            self.assertEqual(file_sha256(result['fixture.bin']),hashlib.sha256(self.payload).hexdigest())

if __name__=='__main__':unittest.main()
