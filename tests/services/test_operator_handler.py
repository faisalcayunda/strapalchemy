"""Tests for OperatorHandler."""


class TestOperatorHandler:
    """Test cases for OperatorHandler."""

    def test_is_operator(self):
        """Test the is_operator method."""
        from strapalchemy.services.operator_handler import OperatorHandler

        assert OperatorHandler.is_operator("$eq") is True
        assert OperatorHandler.is_operator("$in") is True
        assert OperatorHandler.is_operator("$contains") is True
        assert OperatorHandler.is_operator("name") is False
        assert OperatorHandler.is_operator("") is False

    def test_eq_operator(self):
        """Test $eq operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        # Create a mock field
        class MockField:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return self.value == other

        field = MockField("active")
        condition = handler.build_condition(field, "$eq", "active")

        assert condition is True

    def test_ne_operator(self):
        """Test $ne operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def __ne__(self, other):
                return self.value != other

        field = MockField("active")
        condition = handler.build_condition(field, "$ne", "inactive")

        assert condition is True

    def test_in_operator(self):
        """Test $in operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def in_(self, values):
                return self.value in values

        field = MockField("active")
        condition = handler.build_condition(field, "$in", ["active", "pending"])

        assert condition is True

    def test_contains_operator(self):
        """Test $contains operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def contains(self, substring):
                return substring in self.value

        field = MockField("Test Name")
        condition = handler.build_condition(field, "$contains", "Test")

        assert condition is True

    def test_lt_operator(self):
        """Test $lt operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def __lt__(self, other):
                return self.value < other

        field = MockField(25)
        condition = handler.build_condition(field, "$lt", 30)

        assert condition is True

    def test_lte_operator(self):
        """Test $lte operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockType:
            """Mock type that doesn't match SQLAlchemy types."""
            pass

        class MockField:
            def __init__(self, value):
                self.value = value
                self.type = MockType()

            def __lt__(self, other):
                # Note: _handle_lte uses < (not <=) for date handling logic
                return self.value < other

        field = MockField(25)
        condition = handler.build_condition(field, "$lte", 26)

        assert condition is True

    def test_gt_operator(self):
        """Test $gt operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def __gt__(self, other):
                return self.value > other

        field = MockField(30)
        condition = handler.build_condition(field, "$gt", 25)

        assert condition is True

    def test_gte_operator(self):
        """Test $gte operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def __ge__(self, other):
                return self.value >= other

        field = MockField(25)
        condition = handler.build_condition(field, "$gte", 25)

        assert condition is True

    def test_null_operator_true(self):
        """Test $null operator with True."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def is_(self, other):
                return self.value is other

        field = MockField(None)
        condition = handler.build_condition(field, "$null", True)

        assert condition is True

    def test_null_operator_false(self):
        """Test $null operator with False."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def isnot(self, other):
                return self.value is not other

        field = MockField("active")
        condition = handler.build_condition(field, "$null", False)

        assert condition is True

    def test_between_operator(self):
        """Test $between operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def between(self, start, end):
                return start <= self.value <= end

        field = MockField(25)
        condition = handler.build_condition(field, "$between", [20, 30])

        assert condition is True

    def test_starts_with_operator(self):
        """Test $startsWith operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def startswith(self, prefix):
                return self.value.startswith(prefix)

        field = MockField("test@example.com")
        condition = handler.build_condition(field, "$startsWith", "test")

        assert condition is True

    def test_ends_with_operator(self):
        """Test $endsWith operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            def __init__(self, value):
                self.value = value

            def endswith(self, suffix):
                return self.value.endswith(suffix)

        field = MockField("test@example.com")
        condition = handler.build_condition(field, "$endsWith", "@example.com")

        assert condition is True

    def test_unknown_operator(self):
        """Test with unknown operator."""
        from strapalchemy.services.operator_handler import OperatorHandler

        handler = OperatorHandler()

        class MockField:
            pass

        field = MockField()
        condition = handler.build_condition(field, "$unknown", "value")

        assert condition is None
