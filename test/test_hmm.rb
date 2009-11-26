require 'helper'

class TestHmm < Test::Unit::TestCase
	should "create new classifier" do
		model = HMM::Classifier.new
		assert model.class == HMM::Classifier
	end
	
	should "decode using hand-built model" do
		model = HMM::Classifier.new

		# manually build a classifier
		model.o_lex = ["A", "B"]
		model.q_lex = ["X", "Y", "Z"]
		model.a = NArray[[0.8, 0.1, 0.1],
					[0.2, 0.5, 0.3],
					[0.9, 0.1, 0.0]].transpose(1,0)
		model.b = NArray[ [0.2, 0.8],
					[0.7, 0.3],
					[0.9, 0.1]].transpose(1,0)
		model.pi = NArray[0.5, 0.3, 0.2]

		# apply classifier to a sample observation string
		q_star = model.decode(["A","B","A"])
		assert q_star == ["Z", "X", "X"]
	end

end
