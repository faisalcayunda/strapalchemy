"""Tests for ModelSerializer."""


class TestModelSerializer:
    """Test cases for ModelSerializer."""

    def test_serialize_single_model(self, populated_database):
        """Test serializing a single model."""
        from strapalchemy.services.serializer import ModelSerializer

        user = populated_database["users"][0]

        data = ModelSerializer.serialize(user, fields=["id", "name", "email"])

        assert isinstance(data, dict)
        assert data["id"] == user.id
        assert data["name"] == user.name
        assert data["email"] == user.email

    def test_serialize_list_of_models(self, populated_database):
        """Test serializing a list of models."""
        from strapalchemy.services.serializer import ModelSerializer

        users = populated_database["users"]

        data = ModelSerializer.serialize(users, fields=["id", "name"])

        assert isinstance(data, list)
        assert len(data) == 5
        for item in data:
            assert "id" in item
            assert "name" in item

    def test_serialize_all_fields(self, populated_database):
        """Test serializing all fields."""
        from strapalchemy.services.serializer import ModelSerializer

        user = populated_database["users"][0]

        data = ModelSerializer.serialize(user)

        assert isinstance(data, dict)
        assert "id" in data
        assert "name" in data
        assert "email" in data

    def test_serialize_with_dotted_fields(self, populated_database):
        """Test serializing with dotted field notation."""
        from strapalchemy.services.serializer import ModelSerializer

        user = populated_database["users"][0]

        data = ModelSerializer.serialize(user, fields=["id", "name", "organization.slug"])

        assert isinstance(data, dict)
        assert data["id"] == user.id
        assert "organization" in data
        assert "slug" in data["organization"]

    def test_serialize_with_populate_star(self, populated_database):
        """Test serializing with populate='*'."""
        from strapalchemy.services.serializer import ModelSerializer

        user = populated_database["users"][0]

        data = ModelSerializer.serialize(user, populate="*")

        assert isinstance(data, dict)
        # Organization should be populated
        assert "organization" in data

    def test_serialize_with_populate_list(self, populated_database):
        """Test serializing with populate as list."""
        from strapalchemy.services.serializer import ModelSerializer

        user = populated_database["users"][0]

        data = ModelSerializer.serialize(user, populate=["organization"])

        assert isinstance(data, dict)
        assert "organization" in data

    def test_serialize_none(self):
        """Test serializing None."""
        from strapalchemy.services.serializer import ModelSerializer

        data = ModelSerializer.serialize(None)

        assert data == {}

    def test_serialize_empty_list(self):
        """Test serializing empty list."""
        from strapalchemy.services.serializer import ModelSerializer

        data = ModelSerializer.serialize([])

        assert data == []

    def test_serialize_datetime(self, populated_database):
        """Test that datetime fields are serialized to ISO format."""
        from strapalchemy.services.serializer import ModelSerializer

        user = populated_database["users"][0]

        data = ModelSerializer.serialize(user, fields=["id", "created_at"])

        assert isinstance(data, dict)
        assert "created_at" in data
        # Datetime should be serialized to ISO format string
        if data["created_at"] is not None:
            assert isinstance(data["created_at"], str)

    def test_serialize_with_nested_populate(self, populated_database):
        """Test serializing with nested relationship populate."""
        from strapalchemy.services.serializer import ModelSerializer

        users = populated_database["users"]

        data = ModelSerializer.serialize(users, populate="organization")

        assert isinstance(data, list)
        for item in data:
            if item.get("organization"):
                assert isinstance(item["organization"], dict)
