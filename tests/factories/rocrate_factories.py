import tempfile
from dataclasses import dataclass
from pathlib import Path

import factory
from factory.faker import faker
from rocrate.model.creativework import CreativeWork
from rocrate.model.person import Person
from rocrate.rocrate import ROCrate

fake = faker.Faker()


def fake_orcid_id() -> str:
    """
    Generate a fake ORCID-like identifier.

    This follows the 0000-0000-0000-0000 pattern, but does NOT
    implement the real MOD 11-2 check digit (not needed for tests).
    """
    digits = "".join(str(fake.random_int(0, 9)) for _ in range(16))
    return "https://orcid.org/" + "-".join(digits[i : i + 4] for i in range(0, 16, 4))


class PersonFactory(factory.DictFactory):
    """
    Internal representation of a person, easy to work with in Python.

    We then convert these dicts into RO-Crate Person JSON (@id, @type, ...).
    """

    _id = factory.LazyFunction(fake_orcid_id)
    family_name = factory.LazyFunction(fake.last_name)
    given_name = factory.LazyFunction(fake.first_name)
    affiliation = factory.LazyFunction(lambda: [fake.company()])

    @factory.lazy_attribute
    def name(self) -> str:
        return f"{self.given_name} {self.family_name}"

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        data = super()._create(model_class, *args, **kwargs)
        return {
            "@id": data["_id"],
            "@type": "Person",
            "name": data["name"],
            "familyName": data["family_name"],
            "givenName": data["given_name"],
            "affiliation": data["affiliation"],
        }
    


@dataclass
class TestROCrate:
    crate: ROCrate
    root: Path


class ROCrateFactory(factory.Factory):
    """
    Central factory for a small but valid RO-Crate, usable across tests.

    By default it:
    - creates a temp dir as crate root
    - initialises a fresh ROCrate()
    - adds a tiny data file
    - optionally writes the crate to disk
    """

    class Meta:
        model = TestROCrate

    @staticmethod
    def _create_root_dir() -> Path:
        """Create a temporary directory for this crate."""
        tmp = tempfile.mkdtemp(prefix="rocrate_mcp_")
        return Path(tmp)

    @classmethod
    def _create(cls, model_class, *args, **kwargs) -> TestROCrate:  # type: ignore[override]
        # Extract factory fields
        name: str = kwargs.get("name")
        description: str = kwargs.get("description")
        root: Path = kwargs.pop("root", cls._create_root_dir())

        # 1) Build an empty RO-Crate in memory
        crate = ROCrate()

        # 2) Set some basic root Dataset metadata
        #    (Exact API depends on ro-crate-py version; adjust if needed.)
        root_dataset = crate.root_dataset
        root_dataset["name"] = name or fake.sentence()
        root_dataset["description"] = description or " ".join(
            fake.sentences(fake.random_digit_above_two())
        )

        fake_person = PersonFactory()
        person = crate.add(Person(crate, fake_person.pop("@id"), properties=fake_person))
        license = crate.add(CreativeWork(crate, fake.url(), properties={"name": fake.sentence()}))
        root_dataset["author"] = person
        root_dataset['license'] = license    

        # 3) Create a tiny payload file on disk and add it to the crate
        data_file = root / fake.file_name(extension="txt")
        data_file.write_text(fake.sentence()+"\n", encoding="utf-8")

        # ro-crate-py API: crate.add_file(path, properties=...) is standard.
        crate.add_file(
            str(data_file),
            properties={
                "name": "Test data file",
                "encodingFormat": "text/plain",
            },
        )

        # 4) Persist crate structure to disk (if your ro-crate version supports .write)
        #    If your version only has .write_zip, adapt to that instead.
        #    (I’m not 100% sure of your installed ro-crate-py API here; check once.)
        try:
            crate.write(str(root))
        except AttributeError:
            # Fallback: if only write_zip is available, write a zip to the root
            # and let tests that need on-disk crates use that.
            zip_out = root / "crate.zip"
            crate.write_zip(str(zip_out))

        # 5) Return the wrapper object
        return model_class(crate=crate, root=root)
