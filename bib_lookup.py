"""
A useful tool for looking up Bib entries.

It is an updated version of
https://github.com/wenh06/utils/blob/master/utils_universal/utils_bib.py

"""

import re, warnings, calendar
from time import strptime
from collections import OrderedDict
from typing import Union, Optional, Tuple, List, Sequence, Tuple, Dict, NoReturn
from numbers import Number

import requests
import feedparser


__all__ = [
    "BibLookup",
]


class BibLookup(object):
    """finished, continuous improving,

    References
    ----------
    [1] https://github.com/davidagraf/doi2bib2
    [2] https://arxiv.org/help/api
    [3] https://github.com/mfcovington/pubmed-lookup/
    [4] https://serpapi.com/google-scholar-cite-api
    [5] https://www.bibtex.com/

    Example
    -------
    >>> bl = BibLookup(align="middle")
    >>> res = bl("1707.07183")
    @article{wen2017_1707.07183v2,
       author = {Hao Wen and Chunhui Liu},
        title = {Counting Multiplicities in a Hypersurface over a Number Field},
      journal = {arXiv preprint arXiv:1707.07183v2}
         year = {2017},
        month = {7},
    }
    >>> bl("10.23919/cinc53138.2021.9662801", align="left-middle")
    @inproceedings{Wen_2021,
      author    = {Hao Wen and Jingsu Kang},
      title     = {Hybrid Arrhythmia Detection on Varying-Dimensional Electrocardiography: Combining Deep Neural Networks and Clinical Rules},
      booktitle = {2021 Computing in Cardiology ({CinC})}
      doi       = {10.23919/cinc53138.2021.9662801},
      year      = {2021},
      month     = {9},
      publisher = {{IEEE}},
    }

    TODO:
    use eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi for PubMed, as in [3];
    try using google scholar api described in [4] (unfortunately [4] is charged);
    use `Flask` to write a simple browser-based UI;
    """

    __name__ = "BibLookup"

    def __init__(
        self,
        align: str = "middle",
        ignore_fields: Sequence[str] = ["url"],
        email: Optional[str] = None,
        **kwargs,
    ) -> NoReturn:
        """finished, checked,

        Parameters
        ----------
        align: str, default "middle",
            alignment of the final output, case insensitive,
            can be one of "middle", "left", "left-middle", "left_middle"
        ignore_fields: sequence of str, default ["url"],
            fields to be ignored in the final output,
            case insensitive,
        email: str, optional,
            email for querying PubMed publications
        kwargs: additional key word arguments, including
            "verbose": int,
                default 0,
            "odering": sequence of str,
                default ["author", "title", "journal", "booktitle"],
                case insensitive,

        """
        self.align = align.lower()
        assert self.align in [
            "middle",
            "left",
            "left-middle",
            "left_middle",
        ], f"align must be one of 'middle', 'left', 'left-middle', 'left_middle', but got {self.align}"
        self.email = email
        self._ignore_fields = [k.lower() for k in ignore_fields]
        assert self.align in [
            "middle",
            "left",
            "left-middle",
            "left_middle",
        ]
        colon = "[\s]*:[\s]*"
        # NOTE when applying `re.search`, all strings are converted to lower cases
        # DOI examples:
        # "10.7555/JBR.28.20130191" (a counter example that several bib fields are missing)
        self.__doi_pattern_prefix = "doi[\s]*:[\s]*|(?:https?:\/\/)?(?:dx\.)?doi\.org\/"
        self.__doi_pattern = f"^(?:{self.__doi_pattern_prefix})?10\..+\/.+$"
        # PubMed examples:
        # "22331878" or
        # "http://www.ncbi.nlm.nih.gov/pubmed/22331878"
        self.__pmid_pattern_prefix = f"pmid{colon}|pmcid{colon}"  # and pmcid
        # self.__pmid_pattern = f"^(?:{self.__pmid_pattern_prefix})?(?:\d+|pmc\d+(?:\.\d+)?)$"
        self.__pmurl_pattern_prefix = "(?:https?:\/\/)?(?:pubmed\.ncbi\.nlm\.nih\.gov\/|www\.ncbi\.nlm\.nih\.gov\/pubmed\/)"
        # self.__pmurl_pattern = f"^(?:{self.__pmurl_pattern_prefix})?(?:\d+|pmc\d+(?:\.\d+)?)(?:\/)?$"
        self.__pm_pattern_prefix = (
            f"{self.__pmurl_pattern_prefix}|{self.__pmid_pattern_prefix}"
        )
        self.__pm_pattern = (
            f"^(?:{self.__pm_pattern_prefix})?(?:\d+|pmc\d+(?:\.\d+)?)(?:\/)?$"
        )
        # arXiv examples:
        # "arXiv:1501.00001v1", "arXiv:cs/0012022"
        self.__arxiv_pattern_prefix = (
            f"((?:(?:(?:https?:\/\/)?arxiv.org\/)?abs\/)|arxiv{colon})"
        )
        self.__arxiv_pattern = (
            f"^(?:{self.__arxiv_pattern_prefix})?(?:[\w\-]+\/\d+|\d+\.\d+(v(\d+))?)$"
        )
        # self.__arxiv_pattern_old = f"^(?:{self.__arxiv_pattern_prefix})?[\w\-]+\/\d+$"
        self.__default_err = "Not Found"

        self.verbose = kwargs.get("verbose", 0)
        self._ordering = kwargs.get(
            "ordering", ["author", "title", "journal", "booktitle"]
        )
        self._ordering = [k.lower() for k in self._ordering]

    def __call__(self, identifier: str, align: Optional[str] = None) -> str:
        """finished, checked,

        Parameters
        ----------
        identifier: str,
            identifier of a publication,
            can be DOI, PMID (or url), PMCID (or url), arXiv id,
        align: str, optional,
            alignment of the final output, case insensitive,
            if specified, `self.align` is ignored

        Returns
        -------
        res: str,
            the final output in the `str` format

        """
        category, feed_content = self._obtain_feed_content(identifier)
        if category == "doi":
            res = self._handle_doi(feed_content)
        elif category == "pm":
            res = self._handle_pm(feed_content)
        elif category == "arxiv":
            res = self._handle_arxiv(feed_content)
        elif category == "error":
            res = self.__default_err

        if res != self.__default_err:
            res = self._align_result(res, align=(align or self.align).lower())
        print(res)

        return res

    def _obtain_feed_content(self, identifier: str) -> Tuple[str, dict]:
        """finished, checked,

        Parameters
        ----------
        identifier: str,
            identifier of a publication,
            can be DOI, PMID (or url), PMCID (or url), arXiv id,

        Returns
        -------
        category: str,
            one of "doi", "pm", "arxiv"
        fc: dict,
            feed content to GET or POST

        """
        idtf = identifier.lower().strip()
        if re.search(self.__doi_pattern, idtf):
            url = (
                "https://doi.org/"
                + re.sub(
                    self.__doi_pattern_prefix,
                    "",
                    idtf,
                ).strip("/")
            )
            fc = {
                "url": url,
                "headers": {"Accept": "application/x-bibtex; charset=utf-8"},
            }
            category = "doi"
        elif re.search(self.__pm_pattern, idtf):
            url = (
                "http://www.pubmedcentral.nih.gov/utils/idconv/v1.0/?format=json&ids="
                + re.sub(
                    self.__pm_pattern_prefix,
                    "",
                    idtf,
                ).strip("/")
            )
            fc = {
                "url": url,
            }
            category = "pm"
        elif re.search(self.__arxiv_pattern, idtf):
            url = (
                "http://export.arxiv.org/api/query?id_list="
                + re.sub(
                    self.__arxiv_pattern_prefix,
                    "",
                    idtf,
                ).strip("/")
            )
            fc = {
                "url": url,
            }
            category = "arxiv"
        else:
            warnings.warn(
                "unrecognized indentifier (none of doi, pmid, pmcid, pmurl, arxiv)"
            )
            category, fc = "error", {}
        if self.verbose > 1:
            print(f"category = {category}")
            print(f"feed content = {fc}")
        return category, fc

    def _handle_doi(self, feed_content: dict) -> str:
        """finished, checked,

        handle a DOI query using POST

        Parameters
        ----------
        feed_content: dict,
            the content to feed to POST

        Returns
        -------
        res: str,
            decoded query result

        """
        r = requests.post(**feed_content)
        res = r.content.decode("utf-8")
        if self.verbose > 1:
            print(res)
        return res

    def _handle_pm(self, feed_content: dict) -> str:
        """finished, checked,

        handle a PubMed query using POST

        Parameters
        ----------
        feed_content: dict,
            the content to feed to POST

        Returns
        -------
        res: str,
            decoded query result

        """
        r = requests.post(**feed_content)
        if self.verbose > 1:
            print(r.json())
        mid_res = r.json()["records"][0]
        doi = mid_res.get("doi", "")
        if self.verbose > 1:
            print(f"doi = {doi}")
        if doi:
            _, feed_content = self._obtain_feed_content(doi)
            res = self._handle_doi(feed_content)
        else:
            res = self.__default_err
        return res

    def _handle_arxiv(self, feed_content: dict) -> Union[str, Dict[str, str]]:
        """finished, checked,

        handle a arXiv query using GET

        Parameters
        ----------
        feed_content: dict,
            the content to feed to GET

        Returns
        -------
        res: dict,
            decoded and parsed query result

        """
        r = requests.get(**feed_content)
        parsed = feedparser.parse(r.content.decode("utf-8")).entries[0]
        if self.verbose > 1:
            print(parsed)
        title = re.sub("[\s]+", " ", parsed["title"])  # sometimes this field has "\n"
        if title == "Error":
            res = self.__default_err
            return res
        arxiv_id = parsed["id"].split("arxiv.org/abs/")[-1]
        year = parsed["published_parsed"].tm_year
        res = {"title": title}
        # authors = []
        # for item in parsed["authors"]:
        #     a = item["name"].split(" ")
        #     if len(a) > 1:
        #         a[-2] = a[-2] + ","
        #     authors.append(" ".join(a))
        # it seems that surnames are put in the last position of full names by arXiv
        authors = [item["name"] for item in parsed["authors"]]
        res["author"] = " and ".join(authors)
        res["year"] = year
        res["month"] = parsed["published_parsed"].tm_mon
        res["journal"] = f"arXiv preprint arXiv:{arxiv_id}"
        res[
            "label"
        ] = f"{parsed['authors'][0]['name'].split(' ')[-1].lower()}{year}_{arxiv_id}"
        res["class"] = "article"
        return res

    def _align_result(
        self, res: Union[str, Dict[str, str]], align: Optional[str] = None
    ) -> str:
        """finished, checked,

        Parameters
        ----------
        res: str or dict,
            result obtained via GET or POST
        align: str, optional,
            alignment of the final output, case insensitive,
            if specified, `self.align` is ignored

        Returns
        -------
        new_str: str,
            the aligned bib string

        """
        _align = (align or self.align).lower()
        assert _align in [
            "middle",
            "left",
            "left-middle",
            "left_middle",
        ], f"align must be one of 'middle', 'left', 'left-middle', 'left_middle', but got {_align}"
        if isinstance(res, str):
            lines = [l.strip() for l in res.split("\n") if len(l.strip()) > 0]
            d = OrderedDict()
            header = lines[0]
            for l in lines[1:-1]:
                key, val = l.strip().split("=")
                # convert month from abbreviation to number
                if (
                    key.lower().strip() == "month"
                    and val.strip("\{\}, ").capitalize() in calendar.month_abbr
                ):
                    _val = val.strip("\{\}, ")
                    val = val.replace(_val, str(strptime(_val, "%b").tm_mon))
                d[key.strip()] = self._enclose_braces(val)
        elif isinstance(res, dict):
            header = f"@{res['class']}{{{res['label']},"
            d = OrderedDict()
            tmp = {k.strip(): v for k, v in res.items() if k not in ["class", "label"]}
            for idx, (k, v) in enumerate(tmp.items()):
                # convert month from abbreviation to number
                if (
                    k.lower().strip() == "month"
                    and isinstance(v, str)
                    and v.strip("\{\}, ").capitalize() in calendar.month_abbr
                ):
                    _v = v.strip("\{\}, ")
                    v = v.replace(_v, str(strptime(_v, "%b").tm_mon))
                d[k] = self._enclose_braces(v)
                if idx < len(tmp) - 1:
                    d[k] += ","

        # all field names to lower case,
        # and ignore the fields in the list `self.ignore_fields`
        d = {k.lower(): v for k, v in d.items() if k.lower() not in self.ignore_fields}

        # re-order the fields according to the list `self.ordering`
        _ordering = self.ordering + [k for k in d if k not in self.ordering]
        _ordering = [k for k in _ordering if k in d]
        d = OrderedDict((k, d[k]) for k in _ordering)

        # align the fields
        max_key_len = max([len(k) for k in d.keys()])
        if _align == "middle":
            lines = (
                [header]
                + [f"{' '*(2+max_key_len-len(k))}{k} = {v}" for k, v in d.items()]
                + ["}"]
            )
        elif _align == "left":
            lines = [header] + [f"{' '*2}{k} = {v}" for k, v in d.items()] + ["}"]
        elif _align in [
            "left-middle",
            "left_middle",
        ]:
            lines = (
                [header]
                + [f"{' '*2}{k}{' '*(1+max_key_len-len(k))}= {v}" for k, v in d.items()]
                + ["}"]
            )
        new_str = "\n".join(lines)
        return new_str

    def _enclose_braces(self, s: Union[int, str]) -> str:
        """finished, checked,

        ensure that the input string is enclosed with braces

        Parameters
        ----------
        s: str,
            the input string, possibly enclosed with braces and possibly not

        Returns
        -------
        new_s: str,
            the string `s` enclosed with braces

        """
        s_str = str(s).strip()
        # new_s = s_str.strip("{},")  # counter example "publisher = {{IOP} Publishing},"
        # new_s = f"{{{new_s}}}"
        new_s = s_str.strip(",")
        if not all([new_s.startswith("{"), new_s.endswith("}")]):
            new_s = f"{{{new_s}}}"
        if s_str.endswith(","):
            new_s += ","
        return new_s

    @property
    def doi_pattern(self) -> str:
        return self.__doi_pattern

    @property
    def arxiv_pattern(self) -> str:
        return self.__arxiv_pattern

    @property
    def pm_pattern(self) -> str:
        return self.__pm_pattern

    @property
    def pubmed_pattern(self) -> str:
        return self.__pm_pattern

    @property
    def ignore_fields(self) -> List[str]:
        return self._ignore_fields

    @property
    def ordering(self) -> List[str]:
        return self._ordering

    def debug(self) -> NoReturn:
        self.verbose = 2
