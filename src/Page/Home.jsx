import { useNavigate } from "react-router-dom";

function WelcomePage() {
    const navigate = useNavigate();

    return (
        <div class="Background" style={styles.background_color_size}>

            <div style={styles.container}>

                <div class="picture">
                    <img
                        src="/GraphRAG for Birds.jpg"
                        alt="Ảnh"
                        style={{ width: "400px", height: "auto" }}
                    />

                </div>

                <p style={styles.desc}>
                    Hệ thống tra cứu thông tin loài chim sử dụng GraphRAG,
                    cho phép truy vấn tri thức bằng ngôn ngữ tự nhiên.
                </p>

                <button style={styles.button} onClick={() => navigate("/chat")}>
                    Bắt đầu
                </button>
            </div>
        </div>
    );
}

const styles = {
    container: {
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        background: "rgb(255 255 255)"
    },
    desc: {
        maxWidth: "500px",
        textAlign: "center",
        marginBottom: "20px"
    },
    button: {
        padding: "12px 24px",
        fontSize: "16px",
        cursor: "pointer",
        color: "white",
        border: "none",
        width: "150px",
        backgroundColor: "#67c0da",
        borderRadius: "10px"
    },
    background_color_size: {
        height: "100 %",
        background_color: "#white"

    },
};

export default WelcomePage;
